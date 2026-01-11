#!/bin/bash
set -euo pipefail

# Part 3/3: evaluation
# - patches config_${MACHINE_ID}.yaml temporarily (restored on exit)
# - runs multiple router methods sequentially on the full TEST_DATASETS list

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

MACHINE_ID="${MACHINE_ID:-B}"
CONFIG_PATH="${ROOT_DIR}/config_${MACHINE_ID}.yaml"

TEST_DATASETS_DEFAULT="math mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology magpie_5k_test alpaca_5k_test big_math_5k_test mmlu_test"

# Always run on the full test set by default (no args needed).
# shellcheck disable=SC2206
DATASETS=(${TEST_DATASETS_DEFAULT})

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

# Router methods to evaluate, in order.
# Notes:
# - self_based and logits_based_routers will expand internally into multiple routers.
# - You can add/remove items here without touching Python code.
ROUTERS=(
  "embedding_mlp"
  "probe"
  "self_based"
  "logits_based_routers"
  "trained_deberta"
)

patch_config_for_router() {
  local router="$1"
  local metric_dir="${SRC_DIR}/metric_results/auto/${router}"

  # Reset config to the original contents before patching.
  cp "${BACKUP_PATH}" "${CONFIG_PATH}"

  export ROUTER_TYPE="${router}"
  export METRIC_RESULTS_DIR="${metric_dir}"

  python - <<'PY'
import os

config_path = os.environ["CONFIG_PATH"]
router_type = os.environ["ROUTER_TYPE"]
metric_dir = os.environ["METRIC_RESULTS_DIR"]

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

def set_top_level(key: str, value_yaml: str):
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            lines[i] = f"{key}: {value_yaml}\n"
            return
    lines.append(f"{key}: {value_yaml}\n")

set_in_block("router", "router_type", f"\"{router_type}\"")
set_top_level("metric_results_dir", f"\"{metric_dir}\"")

open(config_path, "w", encoding="utf-8").writelines(lines)
PY
}

cd "${SRC_DIR}"
for r in "${ROUTERS[@]}"; do
  echo "=== eval router=${r} datasets=${#DATASETS[@]} ==="
  patch_config_for_router "${r}"
  python main.py --mode eval --datasets "${DATASETS[@]}"
done

