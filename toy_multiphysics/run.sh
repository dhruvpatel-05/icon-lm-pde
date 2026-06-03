#!/usr/bin/env bash
# Reproduce the toy multi-regime fluid-tracer ICON-LM experiment.
# Uses the active Python/conda environment by default, trains the model,
# and renders the figure.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${USE_VENV:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  PYTHON_BIN="python"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [m for m in ("torch", "numpy", "matplotlib") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing packages: " + ", ".join(missing) + ". Install with: pip install -r requirements.txt")
PY

"${PYTHON_BIN}" -u train.py | tee train.log
"${PYTHON_BIN}" -u make_figure.py
"${PYTHON_BIN}" -u evaluate.py | tee eval.log
echo "Done. figure.pdf is at: $(pwd)/figure.pdf"
