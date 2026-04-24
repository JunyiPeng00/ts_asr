from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

from wesep.modules.target_speaker.mhfa import SSL_BACKEND_MHFA

from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRThinkerCausalLMOutputWithPast,
    Qwen3ASRThinkerForConditionalGeneration,
)

from .modules import PrototypeBankProjector, RoleConstrainedMaskBuilder, TargetAwareRefinementBlock
from .utils import qwen_audio_token_length, reconcile_three_lengths


@dataclass
class TargetAwareEncoderConfig:
    num_role_layers: int = 5
    num_prototypes: int = 8
    mhfa_heads: int = 4
    mhfa_compression_dim: int = 128
    router_hidden_multiplier: int = 2
    refinement_variant: str = "full"
    shared_refinement: bool = False
    memory_dim: int = 256
    expert_rank: int = 64
    num_experts: int = 3
    refinement_layer_strategy: str = "post_role_all"
    enable_overlap_head: bool = False
    overlap_num_classes: int = 4
    overlap_loss_weight: float = 0.10
    enable_target_consistency: bool = False
    target_consistency_weight: float = 0.05
    target_consistency_mode: str = "hybrid"
    target_consistency_temperature: float = 0.07
    target_consistency_detach_target: bool = True
    enable_router_supervision: bool = False
    router_loss_weight: float = 0.02

    @classmethod
    def from_any(cls, value: Optional["TargetAwareEncoderConfig | dict"]) -> "TargetAwareEncoderConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Unsupported ts_config type: {type(value)!r}")


@dataclass
class TargetAwareAudioEncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    overlap_logits: Optional[torch.Tensor] = None
    mix_summary: Optional[torch.Tensor] = None
    router_probs: Optional[torch.Tensor] = None


class TSQwen3ASRAudioEncoder(Qwen3ASRAudioEncoder):
    """Qwen3-ASR audio tower with target-aware role masking and refinement."""

    def __init__(self, config: Qwen3ASRAudioEncoderConfig, ts_config: Optional[TargetAwareEncoderConfig | dict] = None):
        super().__init__(config)
        self.ts_config = TargetAwareEncoderConfig.from_any(ts_config)
        self.config._attn_implementation = "eager"
        self.role_layers = min(self.ts_config.num_role_layers, len(self.layers))
        self.role_mask_builder = RoleConstrainedMaskBuilder()
        self.is_lite = self.ts_config.refinement_variant.startswith("lite")
        self.use_shared_refinement = bool(self.ts_config.shared_refinement)
        if self.ts_config.refinement_variant == "lite_shared":
            self.use_shared_refinement = True
        self.memory_dim = min(self.ts_config.memory_dim, config.d_model) if self.is_lite else config.d_model
        self.refinement_layer_indices = self._build_refinement_layer_indices()
        self.mhfa_backend = SSL_BACKEND_MHFA(
            head_nb=self.ts_config.mhfa_heads,
            feat_dim=config.d_model,
            compression_dim=min(self.ts_config.mhfa_compression_dim, config.d_model),
            embed_dim=self.memory_dim,
            nb_layer=self.role_layers + 1,
        )
        self.prototype_projector = PrototypeBankProjector(
            config.d_model,
            self.ts_config.num_prototypes,
            output_hidden_size=self.memory_dim,
        )
        self.mix_memory_proj = None
        self.memory_to_hidden = None
        if self.is_lite:
            self.mix_memory_proj = torch.nn.Linear(config.d_model, self.memory_dim)
            self.memory_to_hidden = torch.nn.Linear(self.memory_dim, config.d_model)
            torch.nn.init.zeros_(self.memory_to_hidden.weight)
            torch.nn.init.zeros_(self.memory_to_hidden.bias)

        num_blocks = max(1, len(self.refinement_layer_indices)) if self.refinement_layer_indices else 0
        if self.use_shared_refinement and num_blocks > 0:
            self.refinement_blocks = torch.nn.ModuleList([self._make_refinement_block(config)])
        else:
            self.refinement_blocks = torch.nn.ModuleList(
                [self._make_refinement_block(config) for _ in range(num_blocks)]
            )

        self.summary_norm = None
        self.mix_summary_attn = None
        self.mix_summary_proj = None
        if self.ts_config.enable_target_consistency:
            self.summary_norm = torch.nn.LayerNorm(config.d_model)
            self.mix_summary_attn = torch.nn.Linear(config.d_model, 1)
            self.mix_summary_proj = torch.nn.Linear(config.d_model, self.memory_dim)
        self.overlap_head = (
            torch.nn.Linear(config.d_model, self.ts_config.overlap_num_classes)
            if self.ts_config.enable_overlap_head
            else None
        )
        self._last_debug: dict[str, torch.Tensor | int | float | tuple[int, ...]] = {}

    def _build_refinement_layer_indices(self) -> list[int]:
        post_role_layers = list(range(self.role_layers, len(self.layers)))
        if not post_role_layers:
            return []
        if self.ts_config.refinement_layer_strategy == "post_role_all":
            return post_role_layers
        if self.ts_config.refinement_layer_strategy == "sparse_4":
            if len(post_role_layers) <= 4:
                return post_role_layers
            positions = torch.linspace(0, len(post_role_layers) - 1, steps=4)
            picked = sorted({post_role_layers[int(round(pos.item()))] for pos in positions})
            return picked
        raise ValueError(
            "Unsupported refinement_layer_strategy: "
            f"{self.ts_config.refinement_layer_strategy}"
        )

    def _make_refinement_block(self, config: Qwen3ASRAudioEncoderConfig) -> TargetAwareRefinementBlock:
        hidden_size = self.memory_dim if self.is_lite else config.d_model
        expert_hidden_size = self.ts_config.expert_rank if self.is_lite else None
        return TargetAwareRefinementBlock(
            hidden_size=hidden_size,
            router_hidden_multiplier=self.ts_config.router_hidden_multiplier,
            expert_hidden_size=expert_hidden_size,
            num_experts=self.ts_config.num_experts,
        )

    def _encode_frontend(self, input_features: torch.Tensor, feature_len: int) -> torch.Tensor:
        feature = input_features[:, :feature_len]
        chunk_num = max((feature_len + self.n_window * 2 - 1) // (self.n_window * 2), 1)
        chunk_lengths = [self.n_window * 2] * chunk_num
        remainder = feature_len % (self.n_window * 2)
        if remainder != 0:
            chunk_lengths[-1] = remainder

        chunk_list = feature.T.split(chunk_lengths, dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2).unsqueeze(1)
        feature_lens_after_cnn = qwen_audio_token_length(
            torch.tensor(chunk_lengths, dtype=torch.long, device=feature.device)
        )
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(int(length), dtype=torch.bool, device=feature.device) for length in feature_lens_after_cnn.tolist()],
            batch_first=True,
        )

        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        batch_size, channels, freq, time = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(batch_size, time, channels * freq)
        )
        position = self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(
            padded_embed.dtype
        )
        padded_embed = padded_embed + position
        return padded_embed[padded_mask_after_cnn]

    def _resolve_segment_lengths(
        self,
        total_feature_len: int,
        total_token_len: int,
        segment_feature_lengths: torch.Tensor,
        segment_token_lengths: Optional[torch.Tensor],
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        feature_lengths = reconcile_three_lengths(total_feature_len, segment_feature_lengths.tolist())
        if segment_token_lengths is None:
            raw_token_lengths = qwen_audio_token_length(feature_lengths)
            token_lengths = reconcile_three_lengths(total_token_len, raw_token_lengths.tolist())
        else:
            token_lengths = reconcile_three_lengths(total_token_len, segment_token_lengths.tolist())
        return feature_lengths, token_lengths

    def _run_target_summary_stack(self, hidden_states: torch.Tensor) -> torch.Tensor:
        total_token_len = int(hidden_states.size(0))
        device = hidden_states.device
        cu_seqlens = torch.tensor([0, total_token_len], device=device, dtype=torch.int32)
        stacked_states = [hidden_states]
        for layer_idx in range(self.role_layers):
            hidden_states = self.layers[layer_idx](
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=None,
            )[0]
            stacked_states.append(hidden_states)
        target_stack = torch.stack(stacked_states, dim=-1).permute(1, 0, 2).unsqueeze(0)
        return self.mhfa_backend(target_stack)

    def encode_target_summary(
        self,
        input_features: torch.Tensor,
        feature_len: torch.LongTensor,
    ) -> torch.Tensor:
        resolved_feature_len = int(feature_len.reshape(-1)[0].item())
        hidden_states = self._encode_frontend(input_features, resolved_feature_len)
        if hidden_states.size(0) == 0:
            raise ValueError("Target summary encoding requires non-empty target features")
        return self._run_target_summary_stack(hidden_states)

    def _pool_mix_summary(self, mix_hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.summary_norm is None or self.mix_summary_attn is None or self.mix_summary_proj is None:
            return None
        normalized = self.summary_norm(mix_hidden_states)
        attn_logits = self.mix_summary_attn(normalized).transpose(0, 1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = torch.matmul(attn_weights, normalized).squeeze(0)
        return self.mix_summary_proj(pooled).unsqueeze(0)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.LongTensor] = None,
        aftercnn_lens: Optional[torch.LongTensor] = None,
        segment_feature_lengths: Optional[torch.LongTensor] = None,
        segment_token_lengths: Optional[torch.LongTensor] = None,
    ) -> TargetAwareAudioEncoderOutput:
        del aftercnn_lens
        if feature_lens is None:
            raise ValueError("feature_lens is required for TSQwen3ASRAudioEncoder")
        if segment_feature_lengths is None:
            raise ValueError("segment_feature_lengths is required for TSQwen3ASRAudioEncoder")

        feature_len = int(feature_lens.reshape(-1)[0].item())
        hidden_states = self._encode_frontend(input_features, feature_len)
        total_token_len = int(hidden_states.size(0))
        device = hidden_states.device
        dtype = hidden_states.dtype

        seg_feat = segment_feature_lengths.reshape(-1, 3)[0].to(device=device, dtype=torch.long)
        seg_tok = None
        if segment_token_lengths is not None:
            seg_tok = segment_token_lengths.reshape(-1, 3)[0].to(device=device, dtype=torch.long)
        feature_lengths, token_lengths = self._resolve_segment_lengths(feature_len, total_token_len, seg_feat, seg_tok)

        enroll_tokens = int(token_lengths[0].item())
        silence_tokens = int(token_lengths[1].item())
        mix_tokens = int(token_lengths[2].item())
        if total_token_len == 0 or mix_tokens <= 0:
            raise ValueError(
                "TSQwen3ASRAudioEncoder requires positive mixture token length, "
                f"got total={total_token_len}, mix={mix_tokens}"
            )

        if enroll_tokens <= 0:
            enroll_tokens = min(1, total_token_len)
            token_lengths[0] = enroll_tokens
            token_lengths[2] = max(total_token_len - enroll_tokens - silence_tokens, 1)
            mix_tokens = int(token_lengths[2].item())

        mix_start = min(total_token_len, enroll_tokens + silence_tokens)
        mix_end = min(total_token_len, mix_start + mix_tokens)

        role_mask = self.role_mask_builder(
            seq_len=total_token_len,
            enroll_len=enroll_tokens,
            silence_len=silence_tokens,
            device=device,
            dtype=dtype,
        )
        cu_seqlens = torch.tensor([0, total_token_len], device=device, dtype=torch.int32)

        enroll_states = [hidden_states[:enroll_tokens]]
        for layer_idx in range(self.role_layers):
            hidden_states = self.layers[layer_idx](
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=role_mask,
            )[0]
            enroll_states.append(hidden_states[:enroll_tokens])

        enroll_stack = torch.stack(enroll_states, dim=-1).permute(1, 0, 2).unsqueeze(0)
        global_anchor = self.mhfa_backend(enroll_stack)
        prototype_bank = self.prototype_projector(hidden_states[:enroll_tokens].unsqueeze(0))

        mix_hidden_states = hidden_states[mix_start:mix_end]
        mix_cu_seqlens = torch.tensor([0, mix_hidden_states.size(0)], device=device, dtype=torch.int32)
        layer_to_block_index = {
            layer_idx: (0 if self.use_shared_refinement else block_idx)
            for block_idx, layer_idx in enumerate(self.refinement_layer_indices)
        }
        last_block_debug = None
        for layer_idx in range(self.role_layers, len(self.layers)):
            mix_hidden_states = self.layers[layer_idx](
                mix_hidden_states,
                cu_seqlens=mix_cu_seqlens,
                attention_mask=None,
            )[0]
            if layer_idx not in layer_to_block_index:
                continue
            block = self.refinement_blocks[layer_to_block_index[layer_idx]]
            if self.is_lite:
                if self.mix_memory_proj is None or self.memory_to_hidden is None:
                    raise RuntimeError("Lite refinement requested but projection layers are missing")
                mix_memory_states = self.mix_memory_proj(mix_hidden_states)
                refined_memory_states = block(
                    mix_memory_states.unsqueeze(0),
                    global_anchor,
                    prototype_bank,
                ).squeeze(0)
                mix_hidden_states = mix_hidden_states + self.memory_to_hidden(
                    refined_memory_states - mix_memory_states
                )
            else:
                mix_hidden_states = block(
                    mix_hidden_states.unsqueeze(0),
                    global_anchor,
                    prototype_bank,
                ).squeeze(0)
            last_block_debug = block.last_debug

        normalized_mix_states = self.ln_post(mix_hidden_states)
        overlap_logits = self.overlap_head(normalized_mix_states) if self.overlap_head is not None else None
        mix_summary = self._pool_mix_summary(normalized_mix_states)

        mix_hidden_states = self.proj1(normalized_mix_states)
        mix_hidden_states = self.act(mix_hidden_states)
        mix_hidden_states = self.proj2(mix_hidden_states)

        router_probs = None
        self._last_debug = {
            "feature_lengths": feature_lengths.detach().cpu(),
            "token_lengths": token_lengths.detach().cpu(),
            "mix_token_start": mix_start,
            "mix_token_end": mix_end,
            "anchor_shape": tuple(global_anchor.shape),
            "prototype_shape": tuple(prototype_bank.shape),
        }
        if overlap_logits is not None:
            self._last_debug["overlap_shape"] = tuple(overlap_logits.shape)
        if mix_summary is not None:
            self._last_debug["mix_summary_shape"] = tuple(mix_summary.shape)
        if last_block_debug is not None:
            last_debug = last_block_debug
            router_probs = last_debug["router_probs"].squeeze(0)
            self._last_debug["router_probs"] = router_probs.detach().cpu()
            self._last_debug["gate_mean"] = float(last_debug["gate_mean"].detach().cpu().item())
            if "retrieve_weights" in last_debug:
                retrieve_weights = last_debug["retrieve_weights"].detach()
                retrieve_entropy = -(
                    retrieve_weights.clamp_min(1.0e-8).log() * retrieve_weights
                ).sum(dim=-1).mean()
                self._last_debug["retrieve_entropy"] = float(retrieve_entropy.detach().cpu().item())
                self._last_debug["retrieve_max_prob"] = float(
                    retrieve_weights.max(dim=-1).values.mean().detach().cpu().item()
                )

        return TargetAwareAudioEncoderOutput(
            last_hidden_state=mix_hidden_states,
            overlap_logits=overlap_logits,
            mix_summary=mix_summary,
            router_probs=router_probs,
        )


class TSQwen3ASRThinkerForConditionalGeneration(Qwen3ASRThinkerForConditionalGeneration):
    def __init__(self, config, ts_config: Optional[TargetAwareEncoderConfig | dict] = None):
        super().__init__(config)
        self.ts_config = TargetAwareEncoderConfig.from_any(ts_config)
        self.config.tsasr = asdict(self.ts_config)
        self.audio_tower = TSQwen3ASRAudioEncoder(config.audio_config, self.ts_config)
        self._last_aux_logs: dict[str, float] = {}

    @classmethod
    def from_base_thinker(
        cls,
        base_thinker: Qwen3ASRThinkerForConditionalGeneration,
        ts_config: Optional[TargetAwareEncoderConfig | dict] = None,
    ) -> "TSQwen3ASRThinkerForConditionalGeneration":
        ts_thinker = cls(base_thinker.config, ts_config=ts_config)
        state_dict = base_thinker.state_dict()
        ts_thinker.load_state_dict(state_dict, strict=False)
        device = next(base_thinker.parameters()).device
        dtype = next(base_thinker.parameters()).dtype
        ts_thinker.to(device=device, dtype=dtype)
        return ts_thinker

    def _flatten_mixture_labels(
        self,
        labels: Optional[torch.LongTensor],
        audio_feature_lengths: Optional[torch.LongTensor],
    ) -> Optional[torch.LongTensor]:
        if labels is None or audio_feature_lengths is None:
            return None
        chunks = []
        for sample_idx, mix_len in enumerate(audio_feature_lengths.tolist()):
            mix_len = int(mix_len)
            chunks.append(labels[sample_idx, :mix_len])
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def _compute_overlap_loss(
        self,
        overlap_logits: Optional[torch.Tensor],
        overlap_labels: Optional[torch.LongTensor],
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        if overlap_logits is None or overlap_labels is None:
            return None, {}
        valid_mask = overlap_labels != -100
        if not torch.any(valid_mask):
            return None, {}
        valid_logits = overlap_logits[valid_mask]
        valid_labels = overlap_labels[valid_mask]
        loss = F.cross_entropy(valid_logits, valid_labels)
        predictions = valid_logits.argmax(dim=-1)
        accuracy = (predictions == valid_labels).float().mean()
        return loss, {
            "tsasr/overlap_loss": float(loss.detach().cpu().item()),
            "tsasr/overlap_acc": float(accuracy.detach().cpu().item()),
        }

    def _compute_router_loss(
        self,
        router_probs: Optional[torch.Tensor],
        overlap_labels: Optional[torch.LongTensor],
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        if router_probs is None or overlap_labels is None:
            return None, {}
        coarse_labels = overlap_labels.clone()
        coarse_labels = torch.where(coarse_labels == 1, torch.zeros_like(coarse_labels), coarse_labels)
        coarse_labels = torch.where(coarse_labels == 2, torch.ones_like(coarse_labels), coarse_labels)
        coarse_labels = torch.where(coarse_labels == 3, torch.full_like(coarse_labels, 2), coarse_labels)
        coarse_labels = torch.where(coarse_labels == 0, torch.full_like(coarse_labels, -100), coarse_labels)
        valid_mask = coarse_labels != -100
        if not torch.any(valid_mask):
            return None, {}
        valid_probs = router_probs[valid_mask]
        valid_labels = coarse_labels[valid_mask]
        loss = F.nll_loss(valid_probs.clamp_min(1.0e-8).log(), valid_labels)
        mean_probs = valid_probs.mean(dim=0)
        return loss, {
            "tsasr/router_loss": float(loss.detach().cpu().item()),
            "tsasr/router_target_ratio": float(mean_probs[0].detach().cpu().item()),
            "tsasr/router_overlap_ratio": float(mean_probs[1].detach().cpu().item()),
            "tsasr/router_nontarget_ratio": float(mean_probs[2].detach().cpu().item()),
        }

    def _compute_target_consistency_loss(
        self,
        mix_summaries: Optional[torch.Tensor],
        target_summaries: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        if mix_summaries is None or target_summaries is None:
            return None, {}

        if self.ts_config.target_consistency_detach_target:
            target_summaries = target_summaries.detach()

        estimated_unit = F.normalize(mix_summaries, dim=-1)
        target_unit = F.normalize(target_summaries, dim=-1)
        cosine_similarity = F.cosine_similarity(estimated_unit, target_unit, dim=-1)

        loss_terms = []
        if self.ts_config.target_consistency_mode in {"cosine", "hybrid"}:
            loss_terms.append(1.0 - cosine_similarity.mean())

        if (
            self.ts_config.target_consistency_mode in {"infonce", "hybrid"}
            and estimated_unit.size(0) > 1
        ):
            temperature = float(self.ts_config.target_consistency_temperature)
            logits = torch.matmul(estimated_unit, target_unit.transpose(0, 1)) / temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            loss_e2t = F.cross_entropy(logits, labels)
            loss_t2e = F.cross_entropy(logits.transpose(0, 1), labels)
            loss_terms.append(0.5 * (loss_e2t + loss_t2e))

        if not loss_terms:
            return None, {}

        loss = torch.stack(loss_terms).mean()
        return loss, {
            "tsasr/target_consistency_loss": float(loss.detach().cpu().item()),
            "tsasr/target_cosine": float(cosine_similarity.mean().detach().cpu().item()),
        }

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        segment_feature_lengths: Optional[torch.LongTensor] = None,
        segment_token_lengths: Optional[torch.LongTensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        del audio_feature_lengths
        if feature_attention_mask is None:
            raise ValueError("feature_attention_mask is required for TSQwen3ASRThinkerForConditionalGeneration")
        if segment_feature_lengths is None:
            raise ValueError("segment_feature_lengths is required for TSQwen3ASRThinkerForConditionalGeneration")

        feature_lens = feature_attention_mask.sum(-1).long()
        audio_features = []
        overlap_logits = []
        mix_summaries = []
        router_probs = []
        for sample_idx, (input_feature, feature_len) in enumerate(zip(input_features, feature_lens)):
            seg_feat = segment_feature_lengths[sample_idx]
            seg_tok = segment_token_lengths[sample_idx] if segment_token_lengths is not None else None
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
                segment_feature_lengths=seg_feat.unsqueeze(0),
                segment_token_lengths=seg_tok.unsqueeze(0) if seg_tok is not None else None,
            )
            audio_features.append(audio_output.last_hidden_state)
            if audio_output.overlap_logits is not None:
                overlap_logits.append(audio_output.overlap_logits)
            if audio_output.mix_summary is not None:
                mix_summaries.append(audio_output.mix_summary)
            if audio_output.router_probs is not None:
                router_probs.append(audio_output.router_probs)
        return {
            "audio_features": torch.cat(audio_features, dim=0),
            "overlap_logits": torch.cat(overlap_logits, dim=0) if overlap_logits else None,
            "mix_summaries": torch.cat(mix_summaries, dim=0) if mix_summaries else None,
            "router_probs": torch.cat(router_probs, dim=0) if router_probs else None,
        }

    def get_target_summaries(
        self,
        target_input_features: Optional[torch.FloatTensor] = None,
        target_feature_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.Tensor]:
        if target_input_features is None or target_feature_attention_mask is None:
            return None
        feature_lens = target_feature_attention_mask.sum(-1).long()
        target_summaries = []
        for input_feature, feature_len in zip(target_input_features, feature_lens):
            target_summary = self.audio_tower.encode_target_summary(
                input_feature[:, :feature_len],
                feature_len.unsqueeze(0),
            )
            target_summaries.append(target_summary)
        if not target_summaries:
            return None
        return torch.cat(target_summaries, dim=0)

    def _merge_audio_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        audio_features: torch.Tensor,
        audio_feature_lengths: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids are required to merge TS-ASR audio features safely")
        token_mask = input_ids == self.config.audio_token_id
        merged = inputs_embeds.clone()
        if audio_feature_lengths is None:
            audio_feature_lengths = token_mask.sum(dim=1)
        offset = 0
        hidden_size = inputs_embeds.size(-1)
        for batch_idx, feature_count in enumerate(audio_feature_lengths.tolist()):
            feature_count = int(feature_count)
            positions = torch.nonzero(token_mask[batch_idx], as_tuple=False).squeeze(-1)
            if positions.numel() != feature_count:
                raise ValueError(
                    f"Audio placeholder count mismatch for sample {batch_idx}: "
                    f"mask={positions.numel()} vs features={feature_count}"
                )
            next_offset = offset + feature_count
            sample_audio = audio_features[offset:next_offset]
            if sample_audio.numel() != feature_count * hidden_size:
                raise ValueError(
                    f"Flattened audio feature size mismatch for sample {batch_idx}: "
                    f"expected {feature_count * hidden_size}, got {sample_audio.numel()}"
                )
            merged[batch_idx, positions] = sample_audio
            offset = next_offset
        if offset != audio_features.size(0):
            raise ValueError(
                f"Unused audio features after merge: used {offset}, total {audio_features.size(0)}"
            )
        return merged

    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        segment_feature_lengths=None,
        segment_token_lengths=None,
        overlap_labels=None,
        overlap_label_resample_count=None,
        target_waveforms=None,
        target_wave_lengths=None,
        target_input_features=None,
        target_feature_attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> Qwen3ASRThinkerCausalLMOutputWithPast:
        del rope_deltas, target_waveforms, target_wave_lengths
        self._last_aux_logs = {}

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_bundle = {
            "audio_features": None,
            "overlap_logits": None,
            "mix_summaries": None,
            "router_probs": None,
        }
        if input_features is not None:
            audio_bundle = self.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                segment_feature_lengths=segment_feature_lengths,
                segment_token_lengths=segment_token_lengths,
            )
            audio_features = audio_bundle["audio_features"]
            if audio_features is None:
                raise RuntimeError("audio_features must not be None when input_features are provided")
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = self._merge_audio_features(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                audio_features=audio_features,
                audio_feature_lengths=audio_feature_lengths,
            )

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        asr_loss = None
        total_loss = None
        if labels is not None:
            asr_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
            )
            total_loss = asr_loss
            self._last_aux_logs["tsasr/asr_loss"] = float(asr_loss.detach().cpu().item())

        flat_overlap_labels = self._flatten_mixture_labels(overlap_labels, audio_feature_lengths)
        if self.ts_config.enable_overlap_head:
            overlap_loss, overlap_logs = self._compute_overlap_loss(audio_bundle["overlap_logits"], flat_overlap_labels)
            self._last_aux_logs.update(overlap_logs)
            if overlap_loss is not None:
                total_loss = (
                    overlap_loss * self.ts_config.overlap_loss_weight
                    if total_loss is None
                    else total_loss + self.ts_config.overlap_loss_weight * overlap_loss
                )

        if self.ts_config.enable_target_consistency:
            target_summaries = self.get_target_summaries(
                target_input_features=target_input_features,
                target_feature_attention_mask=target_feature_attention_mask,
            )
            if target_summaries is not None:
                target_summaries = target_summaries.to(
                    device=audio_bundle["mix_summaries"].device if audio_bundle["mix_summaries"] is not None else logits.device,
                    dtype=audio_bundle["mix_summaries"].dtype if audio_bundle["mix_summaries"] is not None else logits.dtype,
                )
            target_consistency_loss, target_logs = self._compute_target_consistency_loss(
                audio_bundle["mix_summaries"],
                target_summaries,
            )
            self._last_aux_logs.update(target_logs)
            if target_consistency_loss is not None:
                total_loss = (
                    target_consistency_loss * self.ts_config.target_consistency_weight
                    if total_loss is None
                    else total_loss + self.ts_config.target_consistency_weight * target_consistency_loss
                )

        if self.ts_config.enable_router_supervision:
            router_loss, router_logs = self._compute_router_loss(audio_bundle["router_probs"], flat_overlap_labels)
            self._last_aux_logs.update(router_logs)
            if router_loss is not None:
                total_loss = (
                    router_loss * self.ts_config.router_loss_weight
                    if total_loss is None
                    else total_loss + self.ts_config.router_loss_weight * router_loss
                )

        if overlap_label_resample_count is not None and torch.is_tensor(overlap_label_resample_count):
            self._last_aux_logs["tsasr/overlap_label_resample_count"] = float(
                overlap_label_resample_count.detach().reshape(-1)[0].cpu().item()
            )

        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        segment_feature_lengths=None,
        segment_token_lengths=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )
        model_inputs["segment_feature_lengths"] = segment_feature_lengths
        model_inputs["segment_token_lengths"] = segment_token_lengths
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["segment_feature_lengths"] = None
            model_inputs["segment_token_lengths"] = None
        return model_inputs


def upgrade_qwen3_asr_model(base_model, ts_config: Optional[TargetAwareEncoderConfig | dict] = None):
    base_model.thinker = TSQwen3ASRThinkerForConditionalGeneration.from_base_thinker(
        base_model.thinker,
        ts_config=ts_config,
    )
    return base_model
