#!/usr/bin/env bash
set -euo pipefail

python ilm/official_gpt2_encoder/gen.py \
  --hf-dataset wikimedia/wikipedia \
  --hf-config 20231101.en 20231101.ru 20231101.tr \
  --hf-split train \
  --hf-text-field text \
  -o ilm/official_gpt2_encoder