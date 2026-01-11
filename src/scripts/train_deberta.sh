#!/bin/bash
set -euo pipefail

# Train DeBERTa router (Part: train)
# - patches config_${MACHINE_ID}.yaml temporarily (restored on exit)
# - forces router.router_type="deberta"
#
# Usage:
#   bash train_deberta.sh [DATASETS...]
#
# Default datasets (if none provided):
#   alpaca_5k_train big_math_5k_train mmlu_train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

MACHINE_ID="${MACHINE_ID:-B}"
CONFIG_PATH="${ROOT_DIR}/config_${MACHINE_ID}.yaml"

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("alpaca_5k_train" "big_math_5k_train" "mmlu_train")
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

python - <<'PY'
import os

config_path = os.environ["CONFIG_PATH"]
lines = open(config_path, "r", encoding="utf-8").read().splitlines(True)

def find_block(block: str):
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{block}:"):
            start = i
            break
    if start is None:
        lines.append(f"\n{block}:\n")
        start = len(lines) - 1
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j] and not lines[j].startswith(" ") and lines[j].strip().endswith(":"):
            end = j
            break
    return start, end

def set_in_block(block: str, key: str, value_yaml: str):
    start, end = find_block(block)
    prefix = f"  {key}:"
    for i in range(start + 1, end):
        if lines[i].startswith(prefix):
            lines[i] = f"  {key}: {value_yaml}\n"
            return
    lines.insert(start + 1, f"  {key}: {value_yaml}\n")

set_in_block("router", "router_type", "\"deberta\"")

open(config_path, "w", encoding="utf-8").writelines(lines)
PY

cd "${SRC_DIR}"
python main.py --mode train --datasets "${DATASETS[@]}"

