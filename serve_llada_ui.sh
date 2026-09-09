#!/usr/bin/env bash
# Launch LLaDA-UI with a compatible SGLang installation.
# Usage: bash serve_llada_ui.sh <MODEL_PATH> [PORT] [GPU_LIST]
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SERVER_DIR="$SCRIPT_DIR/inference/sglang_server"

MODEL_PATH=${1:?Usage: bash serve_llada_ui.sh <MODEL_PATH> [PORT] [GPU_LIST]}
PORT=${2:-${PORT:-30000}}
GPU_LIST=${3:-${CUDA_VISIBLE_DEVICES:-0}}
HOST=${SGLANG_HOST:-127.0.0.1}

[[ -d "$MODEL_PATH" ]] || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing $MODEL_PATH/config.json" >&2; exit 1; }

if [[ -z ${SGLANG_DP_SIZE:-} ]]; then
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  SGLANG_DP_SIZE=${#gpu_ids[@]}
fi

# Some hosts misreport the GPU compute capability via nvidia-smi (e.g. 8.9
# while PyTorch correctly reports 9.0); the TVM-FFI JIT reads nvidia-smi by
# default and would then build kernels for the wrong arch (runtime error 209,
# "no kernel image is available"). Default to 9.0; otherwise set this to the
# value from `python3 -c "import torch; print(torch.cuda.get_device_capability())"`.
# After changing it, clear the stale JIT cache (~/.cache/tvm-ffi) — its cache
# key does not include the arch. Set to empty to let TVM-FFI auto-detect.
TVM_FFI_CUDA_ARCH_LIST=${TVM_FFI_CUDA_ARCH_LIST:-9.0}
if [[ -n $TVM_FFI_CUDA_ARCH_LIST ]]; then export TVM_FFI_CUDA_ARCH_LIST; fi

runtime_pythonpath=${PYTHONPATH:-}
# cuDNN workaround: on hosts where cuDNN initialization crashes the vision
# patch-embed Conv2d, the bundled shim disables cuDNN (conv falls back to
# aten; the attention/diffusion kernels do not depend on cuDNN).
# Set SGLANG_DISABLE_CUDNN=0 on healthy hosts.
if [[ ${SGLANG_DISABLE_CUDNN:-1} == 1 ]]; then
  no_cudnn_dir="$SERVER_DIR/_no_cudnn"
  if [[ -d "$no_cudnn_dir" ]]; then
    runtime_pythonpath="$no_cudnn_dir${runtime_pythonpath:+:$runtime_pythonpath}"
  fi
fi
# Patched SGLang source tree with the multimodal diffusion hooks (a stock
# 0.5.8 install lacks them). Defaults to the bundled sglang_src/ when present;
# otherwise the installed sglang package is used (apply patches/ per README).
SGLANG_SRC=${SGLANG_SRC:-$SERVER_DIR/sglang_src}
if [[ -n ${SGLANG_SRC} ]]; then
  if [[ -d "$SGLANG_SRC/sglang" ]]; then
    runtime_pythonpath="$SGLANG_SRC${runtime_pythonpath:+:$runtime_pythonpath}"
  else
    echo "WARN: SGLANG_SRC=$SGLANG_SRC has no sglang package, using the installed one" >&2
  fi
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}
export SGLANG_IMAGE_MAX_PIXELS=${SGLANG_IMAGE_MAX_PIXELS:-12845056}

# bd chat template (renders image parts into vision placeholders; without it
# images are silently dropped). Mandatory — override with SGLANG_CHAT_TEMPLATE.
chat_template=${SGLANG_CHAT_TEMPLATE:-$SERVER_DIR/llada2_bd_chat_template.jinja}
[[ -f "$chat_template" ]] || { echo "Missing chat template: $chat_template" >&2; exit 1; }

echo "Launching LLaDA-UI on $HOST:$PORT with GPUs $GPU_LIST (DP=$SGLANG_DP_SIZE)"
CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONPATH="$runtime_pythonpath" \
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --tp-size 1 \
  --dp-size "$SGLANG_DP_SIZE" \
  --disable-radix-cache \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --max-running-requests 6 \
  --chat-template "$chat_template" \
  --context-length 32768
