#!/bin/bash
# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)

stage=-1
stop_stage=-1

mix_data_path='./Libri2Mix/wav16k/max/'
librispeech_root=/scratch/project_465002316/junyi_data/librispeech/LibriSpeech

data=data
noise_type=clean
num_spk=2
num_utts_per_shard=200
train360_num_utts_per_shard=800
num_threads=16
shard_prefix=shards
shard_dsets="train-100 train-360 dev test"
text_dir=""

script_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
recipe_dir=$(cd -- "${script_dir}/.." && pwd)

. "${recipe_dir}/tools/parse_options.sh" || exit 1

data=$(realpath ${data})
if [ -z "${text_dir}" ]; then
  text_dir="${data}/${noise_type}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare the meta files for the datasets"

  for dataset in dev test train-100 train-360; do
    echo "Preparing files for" $dataset

    # Prepare the meta data for the mixed data
    dataset_path=$mix_data_path/$dataset/mix_${noise_type}
    mkdir -p "${data}"/$noise_type/${dataset}
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print $NF}' |
      awk -v path="${dataset_path}" '{print $1 , path "/" $1 , path "/../s1/" $1 , path "/../s2/" $1}' |
      sed 's#.wav##' | sort -k1,1 >"${data}"/$noise_type/${dataset}/wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/wav.scp |
      awk -F[_-] '{print $0, $1,$4}' >"${data}"/$noise_type/${dataset}/utt2spk

    # Prepare the meta data for single speakers
    dataset_path=$mix_data_path/$dataset/s1
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s1/" $NF, $0}' | sort -k1,1 >"${data}"/$noise_type/${dataset}/single.wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/single.wav.scp | grep 's1' |
      awk -F[-_/] '{print $0, $2}' >"${data}"/$noise_type/${dataset}/single.utt2spk

    dataset_path=$mix_data_path/$dataset/s2
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s2/" $NF, $0}' | sort -k1,1 >>"${data}"/$noise_type/${dataset}/single.wav.scp

    awk '{print $1}' "${data}"/$noise_type/${dataset}/single.wav.scp | grep 's2' |
      awk -F[-_/] '{print $0, $5}' >>"${data}"/$noise_type/${dataset}/single.utt2spk
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Prepare target-speaker enrollment from original LibriSpeech"

  for dset in train-100 train-360; do
    librispeech_subset="${librispeech_root}/train-clean-100"
    if [ "${dset}" = "train-360" ]; then
      librispeech_subset="${librispeech_root}/train-clean-360"
    fi
    python "${script_dir}/prepare_spk2enroll_librispeech.py" \
      "${librispeech_subset}" \
      --is_librimix False \
      --outfile "${data}"/$noise_type/${dset}/spk2enroll.json \
      --audio_format flac
  done

  for dset in dev test; do
    librispeech_subset="${librispeech_root}/dev-clean"
    if [ "${dset}" = "test" ]; then
      librispeech_subset="${librispeech_root}/test-clean"
    fi
    python "${script_dir}/prepare_spk2enroll_librispeech.py" \
      "${librispeech_subset}" \
      --is_librimix False \
      --outfile "${data}"/$noise_type/${dset}/spk2enroll.json \
      --audio_format flac
  done

  for dset in dev test train-100 train-360; do
    tmp_single_wav_scp="${data}/${noise_type}/${dset}/single.wav.scp.librispeech"
    backup_single_wav_scp="${data}/${noise_type}/${dset}/single.wav.scp.librimix_backup"
    if [ -f "${data}/${noise_type}/${dset}/single.wav.scp" ] && [ ! -f "${backup_single_wav_scp}" ]; then
      cp "${data}/${noise_type}/${dset}/single.wav.scp" "${backup_single_wav_scp}"
    fi
    prefer_split="train-clean-100"
    if [ "${dset}" = "train-360" ]; then
      prefer_split="train-clean-360"
    elif [ "${dset}" = "dev" ]; then
      prefer_split="dev-clean"
    elif [ "${dset}" = "test" ]; then
      prefer_split="test-clean"
    fi
    python "${script_dir}/build_single_wav_scp_from_librispeech.py" \
      --input_scp "${backup_single_wav_scp}" \
      --output_scp "${tmp_single_wav_scp}" \
      --librispeech_root "${librispeech_root}" \
      --prefer_split "${prefer_split}"
    mv "${tmp_single_wav_scp}" "${data}/${noise_type}/${dset}/single.wav.scp"
  done

   for dset in dev test; do
     python "${script_dir}/prepare_librimix_enroll.py" \
       "${data}"/$noise_type/${dset}/wav.scp \
       "${data}"/$noise_type/${dset}/spk2enroll.json \
       --mix2enroll "${data}/${noise_type}/${dset}/mixture2enrollment" \
       --num_spk ${num_spk} \
       --train False \
       --output_dir "${data}"/${noise_type}/${dset} \
       --outfile_prefix "spk"
   done

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Convert current LibriMix metadata to shard format"
  for dset in ${shard_dsets}; do
    dset_dir="${data}/${noise_type}/${dset}"
    shard_utts_per_shard="${num_utts_per_shard}"
    if [ "${dset}" = "train-360" ]; then
      shard_utts_per_shard="${train360_num_utts_per_shard}"
    fi
    if [ ! -f "${dset_dir}/wav.scp" ] || [ ! -f "${dset_dir}/utt2spk" ]; then
      echo "Skip ${dset}: missing ${dset_dir}/wav.scp or ${dset_dir}/utt2spk"
      continue
    fi

    text_args=()
    if compgen -G "${dset_dir}/spk*_text" >/dev/null; then
      text_args=(--text_dir "${dset_dir}")
    else
      spk1_text="${text_dir}/libri2mix_clean_${dset}_spk1_text"
      spk2_text="${text_dir}/libri2mix_clean_${dset}_spk2_text"
      if [ ! -f "${spk1_text}" ] || [ ! -f "${spk2_text}" ]; then
        echo "Missing text file for ${dset}: ${spk1_text} or ${spk2_text}" >&2
        exit 1
      fi
      text_args=(--spk1_text "${spk1_text}" --spk2_text "${spk2_text}")
    fi

    rm -rf "${dset_dir}/shards" "${dset_dir}/shard.list"
    python "${script_dir}/make_shard_list_asr.py" \
      --num_utts_per_shard "${shard_utts_per_shard}" \
      --num_threads "${num_threads}" \
      --prefix "${shard_prefix}" \
      --shuffle \
      "${text_args[@]}" \
      "${dset_dir}/wav.scp" \
      "${dset_dir}/utt2spk" \
      "${dset_dir}/shards" \
      "${dset_dir}/shard.list"
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Download the pre-trained speaker encoders (Resnet34 & Ecapa-TDNN512) from wespeaker..."
  mkdir wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.zip
  unzip voxceleb_resnet34.zip -d wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.zip
  unzip voxceleb_ECAPA512.zip -d wespeaker_models
fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#   echo "Prepare the speaker embeddings using wespeaker pretrained models"
#   for dataset in dev test train-100; do
#     mkdir -p "${data}"/$noise_type/${dataset}
#     echo "Preparing files for" $dataset
#     wespeaker --task embedding_kaldi \
#               --wav_scp "${data}"/$noise_type/${dataset}/single.wav.scp \
#               --output_file "${data}"/$noise_type/${dataset}/embed \
#               -p wespeaker_models/voxceleb_resnet34 \
#               -g 0 # GPU idx
#   done
# fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  if [ ! -d "${data}/raw_data/musan" ]; then
    mkdir -p ${data}/raw_data/musan
    #
    echo "Downloading musan.tar.gz ..."
    echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."
    wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${data}/raw_data
    md5=$(md5sum ${data}/raw_data/musan.tar.gz | awk '{print $1}')
    [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1

    echo "Decompress all archives ..."
    tar -xzvf ${data}/raw_data/musan.tar.gz -C ${data}/raw_data

    rm -rf ${data}/raw_data/musan.tar.gz
  fi

  echo "Prepare wav.scp for musan ..."
  mkdir -p ${data}/musan
  find ${data}/raw_data/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp

  # Convert all musan data to LMDB
  echo "conver musan data to LMDB ..."
  python "${recipe_dir}/tools/make_lmdb.py" ${data}/musan/wav.scp ${data}/musan/lmdb
fi
