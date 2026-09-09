#!/usr/bin/env bash
# Add `llada2_vl` to SGLang's multimodal RoPE-index dispatch so checkpoints
# whose text_config.model_type is `llada2_vl` (the washed LLaDA2-VL format)
# receive correct multimodal position ids. Additive and idempotent: the
# existing `llada2_moe_veomni` entry is left untouched, and the patch is
# skipped when already applied.
#
# The change is one line inside MRotaryEmbedding.get_rope_index, in the
# qwen2_5_vl model_type tuple that already lists `llada2_moe_veomni` (marked
# `# XXX`). Without it, a `llada2_vl` checkpoint raises
# `RuntimeError: Unimplemented model type: llada2_vl` the first time an image
# is processed.
#
# Usage: bash apply_rotary_llada2vl.sh [SGLANG_DIR]
#   SGLANG_DIR defaults to the installed `sglang` package directory.
set -euo pipefail
if [[ -n "${1:-}" ]]; then
  SGLANG_DIR="$1"
else
  SGLANG_DIR="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parent)')"
fi
ROTARY="$SGLANG_DIR/srt/layers/rotary_embedding.py"
if [[ ! -f "$ROTARY" ]]; then
  echo "apply_rotary_llada2vl: rotary_embedding.py not found at $ROTARY" >&2
  exit 1
fi
if grep -q '"llada2_vl",' "$ROTARY"; then
  echo "apply_rotary_llada2vl: already patched, skipping ($ROTARY)"
  exit 0
fi
if ! grep -q '"llada2_moe_veomni", # XXX' "$ROTARY"; then
  echo "apply_rotary_llada2vl: anchor '\`llada2_moe_veomni\`, # XXX' not found; SGLang layout changed, patch manually" >&2
  exit 2
fi
# Insert `"llada2_vl",` on its own line right after the anchored # XXX line.
sed -i '/"llada2_moe_veomni", # XXX/a\                        "llada2_vl",' "$ROTARY"
echo "apply_rotary_llada2vl: patched $ROTARY"
