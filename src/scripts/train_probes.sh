#!/bin/bash
set -euo pipefail

# Part 2/3: probe training
# - patches config_${MACHINE_ID}.yaml temporarily (restored on exit)
# - forces router.router_type="probe"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

MACHINE_ID="${MACHINE_ID:-B}"
CONFIG_PATH="${ROOT_DIR}/config_${MACHINE_ID}.yaml"

# Training knobs
MAX_SAMPLES="${MAX_SAMPLES:-12000}"
SAVE_LOSS_HISTORY="${SAVE_LOSS_HISTORY:-0}"  # 1 to enable

# Split args into datasets (before --) and probe types (after --).
# If no args are provided, default to a 3-dataset mixed training set.
DATASETS=()
PROBES=()
SEEN_SEP=0
for arg in "$@"; do
  if [ "$arg" = "--" ]; then
    SEEN_SEP=1
    continue
  fi
  if [ $SEEN_SEP -eq 0 ]; then
    DATASETS+=("$arg")
  else
    PROBES+=("$arg")
  fi
done

if [ ${#DATASETS[@]} -eq 0 ]; then
  # Default: 3-dataset mix for training
  DATASETS=("alpaca_5k_train" "big_math_5k_train" "mmlu_train")
fi

if [ ${#PROBES[@]} -eq 0 ]; then
  # Default: train the DynamicDirichlet probe
  PROBES=("dynamic_dirichlet")
else
  :
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
export MAX_SAMPLES
export SAVE_LOSS_HISTORY

PROBES_JSON="$(python - <<'PY' "${PROBES[@]}"
import json, sys
print(json.dumps(sys.argv[1:]))
PY
)"
export PROBES_JSON

python - <<'PY'
import os
import json

config_path = os.environ["CONFIG_PATH"]
max_samples = int(os.environ["MAX_SAMPLES"])
save_loss = os.environ["SAVE_LOSS_HISTORY"].strip().lower() in {"1","true","yes","y"}
probes = json.loads(os.environ["PROBES_JSON"])

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
    # end at next top-level key
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
    # insert right after block header
    lines.insert(start + 1, f"  {key}: {value_yaml}\n")

set_in_block("router", "router_type", "\"probe\"")
set_in_block("training", "max_samples", str(max_samples))
set_in_block("training", "save_loss_history", "true" if save_loss else "false")
set_in_block("training", "probe_types", json.dumps(probes))

open(config_path, "w", encoding="utf-8").writelines(lines)
PY

cd "${SRC_DIR}"
python main.py --mode train --datasets "${DATASETS[@]}"

