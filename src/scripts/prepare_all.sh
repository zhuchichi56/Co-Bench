#!/bin/bash
set -euo pipefail

# Part 1/3: data preparation
# - default: compute scores + logits + embeddings for a full dataset list
# - patches config_${MACHINE_ID}.yaml temporarily (restored on exit)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

MACHINE_ID="${MACHINE_ID:-B}"
CONFIG_PATH="${ROOT_DIR}/config_${MACHINE_ID}.yaml"

PREPARE_STEPS_JSON='["scores","logits","embeddings"]'
PREPARE_TEXT_FIELD="${PREPARE_TEXT_FIELD:-instruction}"
PREPARE_EMBED_BATCH_SIZE="${PREPARE_EMBED_BATCH_SIZE:-64}"

TEST_DATASETS_DEFAULT="math mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology magpie_5k_test alpaca_5k_test big_math_5k_test mmlu_test"

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  # shellcheck disable=SC2206
  DATASETS=(${TEST_DATASETS_DEFAULT})
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "error: config not found: ${CONFIG_PATH}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_PATH="${CONFIG_PATH}.bak.${TS}"
cp "${CONFIG_PATH}" "${BACKUP_PATH}"
trap 'mv -f "${BACKUP_PATH}" "${CONFIG_PATH}"' EXIT

export MACHINE_ID
export CONFIG_PATH
export PREPARE_STEPS_JSON
export PREPARE_TEXT_FIELD
export PREPARE_EMBED_BATCH_SIZE

python - <<'PY'
import os

config_path = os.environ["CONFIG_PATH"]
steps_json = os.environ["PREPARE_STEPS_JSON"]
text_field = os.environ["PREPARE_TEXT_FIELD"]
embed_bs = os.environ["PREPARE_EMBED_BATCH_SIZE"]

lines = open(config_path, "r", encoding="utf-8").read().splitlines(True)

def set_top_level(key: str, value_yaml: str) -> None:
    global lines
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            lines[i] = f"{key}: {value_yaml}\n"
            return
    lines.append(f"{key}: {value_yaml}\n")

set_top_level("prepare_steps", steps_json)
set_top_level("prepare_text_field", f"\"{text_field}\"")
set_top_level("prepare_embed_batch_size", str(int(embed_bs)))

open(config_path, "w", encoding="utf-8").writelines(lines)
PY

cd "${SRC_DIR}"
python main.py --mode prepare --datasets "${DATASETS[@]}"

