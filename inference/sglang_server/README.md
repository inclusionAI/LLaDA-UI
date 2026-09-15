# SGLang Server Deployment

This directory provides the LLaDA-UI model integration, chat template, and an
optional cuDNN workaround for an OpenAI-compatible SGLang endpoint.

## Requirements

Use a dedicated serving environment. The tested version family is:

| Package | Version |
|---|---|
| SGLang | 0.5.8-compatible |
| FlashInfer | 0.6.2 |
| Transformers | 5.0.0 |
| PyTorch | 2.10.x with a compatible CUDA build |
| nvidia-cutlass-dsl | 4.3.4 |

`nvidia-cutlass-dsl` 4.3.4 is required by FlashInfer for its large-image
kernels; without it, high-resolution (e.g. 4K) requests fall back to a slow
path. Install with `pip install --no-deps nvidia-cutlass-dsl==4.3.4` (4.6
breaks flash-attn). The serving stack needs SGLang's multimodal diffusion hooks. Until a public
SGLang build is pinned, minor source-level compatibility changes may be needed
in addition to the three model files included here.

## Install the model integration

Locate the installed package:

```bash
SGLANG_DIR="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parent)')"
```

From this directory, copy the model files into that package:

```bash
cp patches/python/sglang/srt/models/llada2.py \
  "$SGLANG_DIR/srt/models/llada2.py"
cp patches/python/sglang/srt/models/llada2_vl.py \
  "$SGLANG_DIR/srt/models/llada2_vl.py"
cp patches/python/sglang/srt/multimodal/processors/llada2_vl.py \
  "$SGLANG_DIR/srt/multimodal/processors/llada2_vl.py"
```

Then apply the multimodal RoPE-index dispatch patch so checkpoints whose
`text_config.model_type` is `llada2_vl` (the washed LLaDA2-VL format) receive
correct multimodal position ids. The patch is additive and idempotent:

```bash
bash patches/apply_rotary_llada2vl.sh "$SGLANG_DIR"
```

Reinstalling SGLang restores its original files.

## Launch

The model path must be an unpacked, unfused MoE Hugging Face checkpoint. From
the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 SGLANG_DP_SIZE=2 \
bash serve_llada_ui.sh /path/to/hf_ckpt_unpacked
```

The server binds to `127.0.0.1:30000` by default. Configure it with environment
variables when needed:

| Variable | Purpose |
|---|---|
| `SGLANG_HOST` | Bind address; set `0.0.0.0` only behind suitable access control |
| `PORT` | Server port |
| `SGLANG_DP_SIZE` | Number of independent data-parallel workers |
| `SGLANG_IMAGE_MAX_PIXELS` | Image preprocessing limit; defaults to `12845056` |
| `SGLANG_DISABLE_CUDNN` | Bundled cuDNN workaround; defaults to `1`, set `0` on healthy hosts |
| `TVM_FFI_CUDA_ARCH_LIST` | CUDA arch for the TVM-FFI JIT; defaults to `9.0` (see below) |
| `SGLANG_SRC` | Patched SGLang source tree; defaults to the bundled `sglang_src/` |

The launcher runs in the foreground and leaves process supervision to the
calling environment.

## Verify

From another shell at the repository root:

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000/v1
export SGLANG_MODEL=LLaDA-UI

python3 inference/sglang_client.py --example mobile
python3 inference/sglang_client.py --example desktop
python3 inference/sglang_client.py --example web
```

## Compatibility notes

Depending on the exact SGLang and Transformers revisions, the multimodal
diffusion path may also require these upstream integration points:

- register the `llada2_moe_veomni` (and, for washed checkpoints, `llada2_vl`)
  model type and `LLaDA2MoE_VLForConditionalGeneration` architecture;
- add `llada2_vl` to the `model_type` tuple in
  `MRotaryEmbedding.get_rope_index` so `llada2_vl` checkpoints compute
  multimodal RoPE (`patches/apply_rotary_llada2vl.sh` applies this);
- route multimodal embeddings through the diffusion forward batch;
- register the `llada2-vl` conversation template;
- enable the multimodal diffusion path with the FlashInfer backend; and
- adapt `AutoImageProcessor.register` and `load_mm_data` to their installed API
  signatures.

If TVM-FFI builds kernels for the wrong compute capability (runtime error 209,
"no kernel image is available"), set `TVM_FFI_CUDA_ARCH_LIST` to the value
reported by PyTorch (`torch.cuda.get_device_capability`) and clear the stale
TVM-FFI JIT cache (`~/.cache/tvm-ffi`; the cache key does not include the
arch) before restarting. This happens on hosts where `nvidia-smi` misreports
the compute capability. If vision `Conv2d` fails during cuDNN initialization,
keep the bundled no-cuDNN shim on the `PYTHONPATH`. Both workarounds are
enabled by default in the launcher; on healthy hosts disable them with
`TVM_FFI_CUDA_ARCH_LIST=` (empty) and `SGLANG_DISABLE_CUDNN=0`.
