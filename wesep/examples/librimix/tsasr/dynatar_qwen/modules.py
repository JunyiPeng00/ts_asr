from __future__ import annotations

import math

import torch
import torch.nn as nn


class RoleConstrainedMaskBuilder(nn.Module):
    """Builds a block mask for [enroll ; silence ; mix]."""

    def forward(
        self,
        seq_len: int,
        enroll_len: int,
        silence_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
        context_len = min(seq_len, max(0, enroll_len) + max(0, silence_len))
        if context_len == 0 or context_len >= seq_len:
            return mask

        neginf = torch.finfo(dtype).min
        mask[..., :context_len, context_len:] = neginf
        mask[..., context_len:, :context_len] = neginf
        return mask


class PrototypeBankProjector(nn.Module):
    """Compresss enrollment frames into a small bank of target prototypes."""

    def __init__(self, hidden_size: int, num_prototypes: int, output_hidden_size: int | None = None):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.output_hidden_size = output_hidden_size or hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.assign = nn.Linear(hidden_size, num_prototypes, bias=False)
        self.value = nn.Linear(hidden_size, self.output_hidden_size)
        self.out_norm = nn.LayerNorm(self.output_hidden_size)

    def forward(self, enroll_hidden_states: torch.Tensor) -> torch.Tensor:
        if enroll_hidden_states.ndim != 3:
            raise ValueError(
                "Expected enroll_hidden_states with shape [B, T_enroll, D], "
                f"got {tuple(enroll_hidden_states.shape)}"
            )

        normalized = self.norm(enroll_hidden_states)
        assign_logits = self.assign(normalized).transpose(1, 2)
        assign_weights = torch.softmax(assign_logits, dim=-1)
        values = self.value(normalized)
        prototypes = torch.matmul(assign_weights, values)
        return self.out_norm(prototypes)


class TargetEvidenceRetriever(nn.Module):
    """Retrieves per-frame target evidence from the prototype bank."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size**-0.5

    def forward(
        self,
        mix_hidden_states: torch.Tensor,
        prototype_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.query(mix_hidden_states)
        keys = self.key(prototype_bank)
        values = self.value(prototype_bank)
        scores = torch.matmul(queries, keys.transpose(1, 2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        evidence = torch.matmul(weights, values)
        return evidence, weights


class OverlapAwareRouter(nn.Module):
    """Soft routes frames to clean-target / overlap / non-target experts."""

    def __init__(
        self,
        hidden_size: int,
        hidden_multiplier: int = 2,
        num_experts: int = 3,
        expert_hidden_size: int | None = None,
    ):
        super().__init__()
        router_hidden = expert_hidden_size or (hidden_size * hidden_multiplier)
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size * 3, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size * 3, router_hidden),
                    nn.SiLU(),
                    nn.Linear(router_hidden, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.last_router_probs: torch.Tensor | None = None
        self.last_router_logits: torch.Tensor | None = None

    def forward(self, combined_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(combined_states)
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_outputs = torch.stack([expert(combined_states) for expert in self.experts], dim=-2)
        routed = torch.sum(expert_outputs * router_probs.unsqueeze(-1), dim=-2)
        self.last_router_probs = router_probs.detach()
        self.last_router_logits = router_logits.detach()
        return routed, router_probs


class TargetAwareRefinementBlock(nn.Module):
    """Dynamic target-aware refinement on mixture hidden states."""

    def __init__(
        self,
        hidden_size: int,
        router_hidden_multiplier: int = 2,
        expert_hidden_size: int | None = None,
        num_experts: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mix_norm = nn.LayerNorm(hidden_size)
        self.retriever = TargetEvidenceRetriever(hidden_size)
        self.router = OverlapAwareRouter(
            hidden_size,
            hidden_multiplier=router_hidden_multiplier,
            num_experts=num_experts,
            expert_hidden_size=expert_hidden_size,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_debug: dict[str, torch.Tensor] = {}

    def forward(
        self,
        mix_hidden_states: torch.Tensor,
        global_anchor: torch.Tensor,
        prototype_bank: torch.Tensor,
    ) -> torch.Tensor:
        if mix_hidden_states.ndim != 3:
            raise ValueError(
                f"Expected mix_hidden_states [B, T_mix, D], got {tuple(mix_hidden_states.shape)}"
            )

        normalized_mix = self.mix_norm(mix_hidden_states)
        evidence, retrieve_weights = self.retriever(normalized_mix, prototype_bank)
        expanded_anchor = global_anchor.unsqueeze(1).expand_as(normalized_mix)
        combined = torch.cat([normalized_mix, evidence, expanded_anchor], dim=-1)
        routed_delta, router_probs = self.router(combined)
        gate = torch.sigmoid(self.gate(combined))
        refined_delta = gate * self.out_proj(routed_delta)
        self.last_debug = {
            "retrieve_weights": retrieve_weights.detach(),
            "router_probs": router_probs.detach(),
            "router_logits": self.router.last_router_logits,
            "gate_mean": gate.detach().mean(),
        }
        return mix_hidden_states + refined_delta
