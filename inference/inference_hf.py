#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Hugging Face inference for LLaDA-UI.

The file contains the model definition, visual preprocessing, and block-wise
diffusion generation used for single-image grounding. It requires an unpacked
MoE Hugging Face checkpoint; run ``python inference/inference_hf.py --help``
for the grounding and ScreenSpot-V2 verification options.
"""

# ==================== inline: utils/transformers/modeling_rope_utils.py ====================
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import wraps
from typing import Optional, TypedDict

# from .configuration_utils import PreTrainedConfig
# from .utils import is_torch_available, logging

from transformers import PretrainedConfig
from transformers.utils import is_torch_available, logging

logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


def standardize_rope_params(config, rope_theta: float | dict[str, float] | None = None):
    """
    Helper to standardize the config's rope params field by ensuring the params are defined for each
    later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    layer_types = getattr(config, "layer_types", None)
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)

    # Case 1: one RoPE theat = one RoPE param per model without nesting
    if not isinstance(rope_theta, dict):
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        else:
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_theta = rope_parameters.get("rope_theta") or rope_theta
            rope_parameters.update({"rope_theta": rope_theta, "rope_type": rope_type})
        config.rope_parameters = rope_parameters

    # Case 2: different RoPE for each layer as nested dict
    else:
        rope_parameters_per_layer_type = {}
        for layer_type in layer_types:
            if rope_parameters is None:
                rope_parameters_per_layer_type[layer_type] = {
                    "rope_type": "default",
                    "rope_theta": rope_theta[layer_type],
                }
            else:
                is_field_in_new_format = any(layer_type in rope_parameters for layer_type in layer_types)
                if not is_field_in_new_format:
                    curr_rope_type = rope_parameters.get("rope_type", rope_parameters.get("type"))
                    rope_parameters_per_layer_type[layer_type] = {
                        **rope_parameters,
                        "rope_type": curr_rope_type,
                        "rope_theta": rope_theta[layer_type],
                    }
                else:
                    curr_rope_type = rope_parameters[layer_type].get(
                        "rope_type", rope_parameters[layer_type].get("type")
                    )
                    rope_parameters_per_layer_type[layer_type] = {
                        **rope_parameters[layer_type],
                        "rope_type": curr_rope_type,
                        "rope_theta": rope_theta[layer_type],
                    }
            config.rope_parameters = rope_parameters_per_layer_type


def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

    Args:
        rope_forward (Callable):
            The forward pass of the RoPE implementation.

    Returns:
        The decorated forward pass.
    """

    def longrope_frequency_update(self, position_ids, device, layer_type=None):
        """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
        seq_len = torch.max(position_ids) + 1
        original_max_position_embeddings = getattr(
            self.config, "original_max_position_embeddings", self.config.max_position_embeddings
        )
        if layer_type is None:
            rope_type = self.rope_type
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            rope_type = self.rope_type[layer_type]
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        if seq_len > original_max_position_embeddings:
            if not hasattr(self, f"{layer_type}_long_inv_freq"):
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                long_inv_freq, _ = rope_init_fn(
                    self.config,
                    device,
                    seq_len=original_max_position_embeddings + 1,
                    layer_type=layer_type,
                )
            self.register_buffer(f"{prefix}inv_freq", long_inv_freq, persistent=False)
            setattr(self, f"{prefix}long_inv_freq", long_inv_freq)
        else:
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)

    def dynamic_frequency_update(self, position_ids, device, layer_type=None):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if layer_type is None:
            rope_type = self.rope_type
            max_seq_len_cached = self.max_seq_len_cached
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            rope_type = self.rope_type[layer_type]
            max_seq_len_cached = getattr(self, f"{layer_type}_max_seq_len_cached", self.max_seq_len_cached)
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        if seq_len > max_seq_len_cached:  # growth
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(
                self.config,
                device,
                seq_len=seq_len,
                layer_type=layer_type,
            )
            # TODO joao: may break with compilation
            self.register_buffer(f"{prefix}inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_max_seq_len_cached", seq_len)

        if seq_len < self.original_max_seq_len and max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            original_inv_freq = original_inv_freq.to(device)
            self.register_buffer(f"{prefix}inv_freq", original_inv_freq, persistent=False)
            setattr(self, f"{prefix}original_inv_freq", original_inv_freq)
            setattr(self, f"{layer_type}_max_seq_len_cached", self.original_max_seq_len)

    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
        kwargs = {"layer_type": layer_type} if layer_type is not None else {}
        if "dynamic" in rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device, **kwargs)
        elif rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device, **kwargs)
        return rope_forward(self, x, position_ids, **kwargs)

    return wrapper


def _compute_linear_scaling_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    factor = rope_parameters_dict["factor"]

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The default sequence length used to update the dynamic RoPE at
                inference time
            *   rope_parameters (`dict[str, float]`): The standard RoPE scaling parameters, from which `factor`
                will be accessed. The value of `factor` is used to determine the new base frequency, along with the
                current sequence length (seq_len), the maximum positional embeddings (max_position_embeddings), and the
                computed dimensionality (dim) of the rotary embeddings. If seq_len <= max_position_embeddings, this
                factor has no effect. If seq_len <= max_position_embeddings, this factor effectively stretches the
                context window using an exponent derived from `dim`.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time. If `None` or shorter than
            max_position_embeddings, this value will be overridden by max_position_embeddings.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_parameters
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = rope_parameters_dict["factor"]
    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    if seq_len is None:
        seq_len = max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, max_position_embeddings)

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig,
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The maximum length of the positional embeddings.
            *   rope_parameters (`dict[str, float | int]`): The standard RoPE scaling parameters, from which the following
                keys will be accessed:
                *   `attention_factor` (`float`, *optional*): The scaling factor to be applied to the computed cos/sin.
                    If None, the value is inferred from `factor`, `mscale`, and `mscale_all_dim` as avaialble.
                *   `beta_fast` (`float`, *optional*, defaults to 32): Parameter to set the boundary for extrapolation
                    (only) in the linear ramp function.
                *   `beta_slow` (`float`, *optional*, defaults to 1): Parameter to set the boundary for interpolation
                    (only) in the linear ramp function.
                *   `factor` (`float`, *optional*): The scaling factor applied when interpolating the position IDs to
                    extend the possible context length. Additionally, if `attention_factor` is None, the log of this
                    value is used to compute a value for `attention_factor`, possibly in conjunciton with `mscale` and
                    `mscale_all_dim`, if provided.
                *   `mscale` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale` acts scalar augmenting `log(factor)` when computing the
                    numerator for the inferred value of `attention_factor`. If not provided, `attention_factor` will be
                    calculated based on `factor` only.
                *   `mscale_all_dim` (`float`, *optional*): If `attention_factor` is None and both `mscale` and
                    `mscale_all_dim` are provided, `mscale_all_dim` acts scalar augmenting `log(factor)` when computing
                    the denominator for the inferred value of `attention_factor`. If not provided, `attention_factor`
                    will be calculated based on `factor` only.
                *   `original_max_position_embeddings` (`int`, *optional*): The original max position embeddings used
                    during pretraining. If not provided, the function falls back to `max_position_embeddings`.
                *   `truncate` (`bool`, *optional*): Whether to truncate the correction range.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): If less than 1.0, inverse frequencies
                will be returned for the first fraction of the head_dim.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    factor = rope_parameters_dict["factor"]
    attention_factor = rope_parameters_dict.get("attention_factor")
    mscale = rope_parameters_dict.get("mscale")
    mscale_all_dim = rope_parameters_dict.get("mscale_all_dim")

    # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if "original_max_position_embeddings" in rope_parameters_dict:
        original_max_position_embeddings = rope_parameters_dict["original_max_position_embeddings"]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_parameters_dict.get("beta_fast") or 32
    beta_slow = rope_parameters_dict.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_parameters.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: PretrainedConfig,
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   max_position_embeddings (`int`): The maximum length of the positional embeddings.
            *   original_max_position_embeddings (`int`, *optional*): The original max position embeddings used during
                pretraining. If not provided, defaults to `max_position_embeddings`.
            *   rope_parameters (`dict[str, float]`): The standard RoPE scaling parameters, from which the following keys
                will be accessed:
                *   `attention_factor` (`float`, *optional*): The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, inferred from
                    the value of `factor`.
                *   `factor` (`float`, *optional*): The scaling factor to apply to the RoPE embeddings. If both
                    `max_position_embeddings` and `original_max_position_embeddings` are provided, this value will be
                    overridden s the ratio between those values.
                *   `long_factor` (`float`, *optional*): The scale factor applied when computing the inverse
                    frequencies if `seq_len` is provided and greater than `original_max_position_embeddings`.
                *   `short_factor` (`float`, *optional*): The scale factor applied when computing the inverse
                    frequencies if `seq_len` is None or less-than-or-equal-to `original_max_position_embeddings`.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): If less than 1.0, inverse frequencies
                will be returned for the first fraction of the head_dim.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # TODO (joao): use the new `original_max_position_embeddings` from rope_parameters
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    long_factor = rope_parameters_dict["long_factor"]
    short_factor = rope_parameters_dict["short_factor"]
    factor = rope_parameters_dict.get("factor")
    attention_factor = rope_parameters_dict.get("attention_factor")

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if original_max_position_embeddings := getattr(config, "original_max_position_embeddings", None):
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: PretrainedConfig,
    device: "torch.device",
    seq_len: Optional[int] = None,
    layer_type: Optional[str] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.
            *   rope_parameters (`dict[str, float | int]`): The standard RoPE scaling parameters, from which the following
                keys will be accessed:
                *   `factor` (`float`, *optional*): The scaling factor applied to the inverse frequencies when 1) the
                    wavelength is greater than `low_freq_wavelen` prior to smoothing, and 2) to all inverse frequencies
                    during smoothing.
                *   `high_freq_factor` (`float`): The scale factor used to compute `high_freq_wavelen` and
                    the value for the denominator of the smoothing factor prior to the `low_freq_factor` shift.
                *   `low_freq_factor` (`float`): The scale factor used to compute `low_freq_wavelen` and
                    the shift applied to the numerator and denominator of the smoothing factor.
                    frequencies if `seq_len` is None or less-than-or-equal-to `original_max_position_embeddings`.
                *   `original_max_position_embeddings` (`int`): The original max position embeddings used
                    during pretraining. If not provided, the function falls back to `max_position_embeddings`.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    standardize_rope_params(config)
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    # Gets the default RoPE parameters
    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    factor = rope_parameters_dict["factor"]  # `8` in the original implementation
    low_freq_factor = rope_parameters_dict["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = rope_parameters_dict["high_freq_factor"]  # `4` in the original implementation
    old_context_len = rope_parameters_dict["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this rope_parameters to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}


def _check_received_keys(
    rope_type: str,
    received_keys: set,
    required_keys: set,
    optional_keys: Optional[set] = None,
    ignore_keys: Optional[set] = None,
):
    """Compare the received keys in `config.rope_parameters` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")

    # Some models need to store model-specific keys, and we don't want to throw warning at them
    if ignore_keys is not None:
        received_keys -= ignore_keys

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(
    rope_parameters: dict, config: Optional[PretrainedConfig] = None, ignore_keys: Optional[set] = None
):
    required_keys = {"rope_type", "rope_theta"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(
    rope_parameters: dict, config: Optional[PretrainedConfig] = None, ignore_keys: Optional[set] = None
):
    required_keys = {"rope_type", "factor", "rope_theta"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(
    rope_parameters: dict, config: Optional[PretrainedConfig] = None, ignore_keys: Optional[set] = None
):
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(
    rope_parameters: dict, config: Optional[PretrainedConfig] = None, ignore_keys: Optional[set] = None
):
    required_keys = {"rope_type", "factor", "rope_theta"}
    optional_keys = {
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "original_max_position_embeddings",
        "mscale",
        "mscale_all_dim",
    }
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_parameters.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_parameters.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"`rope_parameters`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_parameters.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"`rope_parameters`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(
            f"`rope_parameters`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )

    # Models should set `config.rope_parameters["original_max_position_embeddings"]` to their original (pre-yarn) context
    # length, with `config.max_position_embeddings` corresponding to their post-yarn context length.
    # However, for BC purposes, we allow the former to be unset.
    original_max_position_embeddings = config.rope_parameters.get("original_max_position_embeddings")
    if original_max_position_embeddings is not None:
        # Double-check: `factor` should be the ratio between the pre-yarn and post-yarn context lengths.
        implicit_factor = config.max_position_embeddings / original_max_position_embeddings
        if implicit_factor != factor:
            logger.warning_once(
                f"The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = {factor}) does not match "
                "the ratio implicitly set by other parameters (implicit factor = "
                "post-yarn context length / pre-yarn context length = "
                "config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = "
                f"{implicit_factor}). Using the explicit factor ({factor}) in YaRN. This may cause unexpected "
                "behaviour in model usage, please correct the 'max_position_embeddings' fields in the model config."
            )
    # No `config.rope_parameters["original_max_position_embeddings"]`. Is `config.max_position_embeddings` the
    # pre-yarn or the post-yarn context length?
    # BC: we assume it is the pre-yarn context length.
    else:
        logger.warning_once(
            "config.rope_parameters['original_max_position_embeddings'], the pre-yarn context length, is unset. We will "
            "**assume** config.max_position_embeddings holds the pre-yarn context length. Some use cases may expect "
            "config.max_position_embeddings to hold the post-yarn context length (pre-yarn context length * "
            "factor) -- we recommend updating both fields for optimal downstream model usage."
        )


def _validate_longrope_parameters(rope_parameters: dict, config: PretrainedConfig, ignore_keys: Optional[set] = None):
    required_keys = {"rope_type", "short_factor", "long_factor", "rope_theta"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_parameters.keys())
    rope_type = rope_parameters["rope_type"]
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    short_factor = rope_parameters.get("short_factor")
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        logger.warning(f"`rope_parameters`'s short_factor field must be a list of numbers, got {short_factor}")
    if len(short_factor) != dim // 2:
        logger.warning(f"`rope_parameters`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_parameters.get("long_factor")
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        logger.warning(f"`rope_parameters`'s long_factor field must be a list of numbers, got {long_factor}")
    if len(long_factor) != dim // 2:
        logger.warning(f"`rope_parameters`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_parameters` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        logger.warning_once(
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures."
        )
    else:
        factor = rope_parameters.get("factor")
        if factor is None:
            logger.warning("Missing required keys in `rope_parameters`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None:
            if not isinstance(attention_factor, float) or attention_factor < 0.0:
                logger.warning(
                    f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
                )


def _validate_llama3_parameters(rope_parameters: dict, config: PretrainedConfig, ignore_keys: Optional[set] = None):
    required_keys = {
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
        "rope_theta",
    }
    rope_type = rope_parameters["rope_type"]
    received_keys = set(rope_parameters.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_parameters["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_parameters["low_freq_factor"]
    high_freq_factor = rope_parameters["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        logger.warning(f"`rope_parameters`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        logger.warning(f"`rope_parameters`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor <= low_freq_factor:
        logger.warning(
            "`rope_parameters`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        logger.warning(
            "`rope_parameters`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= config.max_position_embeddings:
        logger.warning(
            "`rope_parameters`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
        )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}


def rope_config_validation(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    """
    Validate the RoPE config arguments, given a `PretrainedConfig` object
    """
    rope_parameters_dict = getattr(config, "rope_parameters", None)  # not a default parameter in `PretrainedConfig`
    if rope_parameters_dict is None:
        return

    if getattr(config, "layer_types", None) is not None and all(
        key in config.layer_types for key in rope_parameters_dict.keys()
    ):
        pass
    else:
        rope_parameters_dict = {"full_attention": rope_parameters_dict}

    for rope_parameters in rope_parameters_dict.values():
        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)

        rope_parameters["rope_type"] = rope_type
        # BC: "rope_theta" was originally saved in config
        rope_parameters["rope_theta"] = rope_parameters.get("rope_theta", getattr(config, "rope_theta", None))

        if validation_fn is not None:
            validation_fn(rope_parameters, config=config, ignore_keys=ignore_keys)
        else:
            logger.warning(
                f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
            )


class RopeParameters(TypedDict):
    """
    Args:
        rope_theta (`float`):
            The base period of the RoPE embeddings.
        rope_type (`str`, *optional*, defaults to "default"):
            The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
            'llama3'], with 'default' being the original RoPE implementation.
        factor (`float`, *optional*):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        original_max_position_embeddings (`int`, *optional*):
            Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
            pretraining.
        attention_factor (`float`, *optional*):
            Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        short_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to short contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        long_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to long contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        low_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
        high_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
    """

    rope_theta: float
    rope_type: Optional[str]
    factor: Optional[float]
    original_max_position_embeddings: Optional[int]
    attention_factor: Optional[float]
    beta_fast: Optional[float]
    beta_slow: Optional[float]
    short_factor: Optional[list[float]]
    long_factor: Optional[list[float]]
    low_freq_factor: Optional[float]
    high_freq_factor: Optional[float]
# ==================== inline: models/llada2_vl/masking_utils.py ====================
# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
# from transformers import PreTrainedConfig
from transformers import PretrainedConfig
from transformers.utils import is_torch_xpu_available, logging
# from transformers.utils.generic import GeneralInterface
from transformers.utils.import_utils import is_torch_flex_attn_available, is_torch_greater_or_equal, is_torchdynamo_compiling


from collections.abc import MutableMapping

class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())



if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor



# _is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
# _is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5")
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6")
_is_torch_xpu_available = is_torch_xpu_available()

if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex


logger = logging.get_logger(__name__)


def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a full bidirectional mask.
    """
    return q_idx.new_ones((), dtype=torch.bool)


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask


def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a chunked attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (kv_idx - left_padding[batch_idx]) // chunk_size == (q_idx - left_padding[batch_idx]) // chunk_size

    return inner_mask


def _legacy_chunked_overlay(chunk_size: int) -> Callable:
    """
    Same as the above function, but do not correctly account for left padding tokens.
    Only kept for compatibility with older torch versions (< 2.6).
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx // chunk_size == q_idx // chunk_size

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def chunked_causal_mask_function(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This return the mask_function function to create a chunked attention mask.
    """
    if not _is_torch_greater_or_equal_than_2_6:
        return and_masks(_legacy_chunked_overlay(chunk_size), causal_mask_function)
    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask


def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, q_idx + q_offset, kv_idx + kv_offset)

    return inner_mask


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs. Optionally, vmap over
    the batch and head indices as well if `bh_indices=True`.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).

    Args:
        mask_function (`Callable`):
            The mask_function to vmap.
        bh_indices (`bool`, optional):
            Whether to vmap over the batch and head indices as well, or only q and kv indices.

    Returns:
        Callable: The vmapped function.
    """
    # We vmap the function 2 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        # We extend broadcasting over the [batch_idx, head_idx] dimensions
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necessary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        # For flex, we should not slice them, only use an offset
        if _slice:
            # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
            # but without data-dependent slicing (i.e. torch.compile friendly)
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    # When using `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
    # hard-coded to the forward. If a user exports a model with query_length > 1, the exported model will hard-code `is_causal=True`
    # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108). Thus, we only set
    # `ignore_causal_mask = True` if we are not tracing
    if (
        not is_tracing
        # only cases when lower and upper diags are the same, see https://github.com/pytorch/pytorch/issues/108108
        and (query_length == 1 or (kv_length == query_length or _is_torch_xpu_available))
        # in this case we need to add special patterns to the mask so cannot be skipped otherwise
        and (local_attention_size is None or kv_length < local_attention_size)
        # In this case, we need to add padding to the mask, so cannot be skipped otherwise
        and (
            padding_mask is None
            or (
                padding_mask.all()
                if not _is_torch_xpu_available or query_length == 1
                else padding_mask[:, :query_length].all()
            )
        )
    ):
        return True

    return False


def _ignore_bidirectional_mask_sdpa(padding_mask: Optional[torch.Tensor]) -> bool:
    """
    Detects whether the bidirectional mask can be ignored in case PyTorch's SDPA is used, i.e. when there is full
    attention with no padding.
    """
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()

    # When using `torch.export` or `torch.onnx.dynamo_export`, we need to avoid to check the contents of the mask;
    # otherwise, we will encounter dynamic control flows
    if not is_tracing and (padding_mask is None or padding_mask.all()):
        return True

    return False


def sdpa_mask_recent_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    This function can only be used with torch>=2.5, as the context manager is otherwise not available.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.


    ## Creating a simple causal mask:

    To create the following causal mask:

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ■ ■ ■ ■ ⬚
        4 ■ ■ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5)
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [ True,  True,  True,  True, False],
                  [ True,  True,  True,  True,  True]]]])
    ```

    ## Creating a sliding window mask:

    To create the following sliding window mask (`sliding_window=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ■ ■ ■ ⬚
        4 ⬚ ⬚ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=sliding_window_causal_mask_function(3))
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [False,  True,  True,  True, False],
                  [False, False,  True,  True,  True]]]])
    ```

    ## Creating a chunked attention mask

    To create the following chunked attention mask (`chunk_size=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ⬚ ⬚ ■ ⬚
        4 ⬚ ⬚ ⬚ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=chunked_causal_mask_function(3, torch.zeros(1, dtype=int)))
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    # Under specific conditions, we can avoid materializing the mask
    #   1. Causal masks can rely on the `is_causal` argument
    #   2. Bidirectional do not need any further processing (no bias)
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None
    if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask):
        return None

    # vmap can incur performance issues as reported in #41566 for bidirectional mask as we only need to expand the
    # padding mask. Thus, we allow early exit here if we do not detect any modification to the base mask function
    if mask_function is bidirectional_mask_function:
        if padding_mask is not None:
            # used for slicing without data-dependent slicing
            mask_indices = torch.arange(kv_length, device=cache_position.device) + kv_offset
            return padding_mask[:, None, None, mask_indices].expand(-1, -1, q_length, -1)
        else:
            return torch.ones(batch_size, 1, q_length, kv_length, dtype=torch.bool, device=cache_position.device)

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=cache_position.device)
    head_arange = torch.arange(1, device=cache_position.device)
    # This creates the 4D mask easily. Note that we need this context manager as vmap cannot handle slicing a tensor from
    # scalar tensor (it internally calls `.item()` which vmap does not allow, but this context works around it
    # We don't need to add an offset to the mask_function either, as we vmap directly the correct indices for k and kv indices
    with TransformGetItemToIndex():
        causal_mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    return causal_mask


def sdpa_mask_older_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    allow_is_bidirectional_skip: bool = False,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    NOTE: This function is only used when torch version is torch<2.5 - see `sdpa_mask_recent_torch` otherwise.

    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    If `allow_torch_fix=True` (the default), rows corresponding to query tokens that do not attend
    to any other tokens (due to padding) will be fully attended to instead, in order to avoid `nan` propagation (this does
    not change the final result).

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.
    """
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask
    #   1. Causal masks can rely on the `is_causal` argument
    #   2. Bidirectional do not need any further processing (no bias)
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None
    if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask):
        return None

    # vmap can incur performance issues as reported in #41566 for bidirectional mask as we only need to expand the
    # padding mask. Thus, we allow early exit here if we do not detect any modification to the base mask function
    if mask_function is bidirectional_mask_function:
        if padding_mask is not None:
            return padding_mask[:, None, None, :].expand(-1, -1, q_length, -1)
        else:
            return torch.ones(batch_size, 1, q_length, kv_length, dtype=torch.bool, device=cache_position.device)

    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    # This creates the 4D mask easily. Note that we do not include vmap over the batch_idx dimension as well,
    # as vmap cannot handle slicing a tensor from scalar tensor (it internally calls `.item()` which vmap does not allow
    # However, in more recent version of Pytorch, a trick was introduced to handle it - which is the reason we have
    # `sdpa_mask_recent_torch`, as it allows more general `mask_function`
    causal_mask = _vmap_for_bhqkv(mask_function, bh_indices=False)(None, None, cache_position, kv_arange)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]

    # Due to a bug in versions of torch<2.5, we need to update the mask in case a query is not attending to any
    # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        causal_mask |= torch.all(~causal_mask, dim=-1, keepdim=True)
    return causal_mask


# We use the version with newer torch whenever possible, as it is more general and can handle arbitrary mask functions
# (especially mask_function indexing a tensor, such as the padding mask function)
sdpa_mask = sdpa_mask_recent_torch if _is_torch_greater_or_equal_than_2_6 else sdpa_mask_older_torch


def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        dtype (`torch.dtype`, optional):
            The dtype to use for the mask. By default, `torch.float32`.
    """
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    _ = kwargs.pop("allow_is_causal_skip", None)
    _ = kwargs.pop("allow_is_bidirectional_skip", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=False,
        allow_torch_fix=False,
        **kwargs,
    )
    min_dtype = torch.finfo(dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
    return mask


def flash_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Create the attention mask necessary to use FA2. Since FA2 is un-padded by definition, here we simply return
    `None` if the mask is fully causal, or we return the 2D mask which will then be used to extract the seq_lens.
    We just slice it in case of sliding window.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    if attention_mask is not None:
        # Here we need to slice from the right if using sliding or chunked (for full attention, this is equivalent to doing nothing)
        attention_mask = attention_mask[:, -kv_length:]
        # We only return an actual mask if there is at least 1 padding token, otherwise we return `None` and use `is_causal` in FA2
        # (note that the attention_mask is a boolean dtype here)
        if attention_mask.all():
            attention_mask = None

    return attention_mask


def flex_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> BlockMask:
    """
    Create a 4D block mask which is a compressed representation of the full 4D block causal mask. BlockMask is essential
    for performant computation of flex attention. See: https://pytorch.org/blog/flexattention/

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
    """
    q_length, q_offset = cache_position.shape[0], cache_position[0]

    # Potentially add the padding 2D mask
    if attention_mask is not None:
        # Older torch (2.5.x) cannot handle sequences not in multiples of 128 (default block size)
        # Hence we pad to multiples of this as a minimum to ensure this
        pad_len = ((attention_mask.shape[1] // flex_default_block_size) + 1) * flex_default_block_size
        pad_len = pad_len - attention_mask.shape[1]
        if not _is_torch_greater_or_equal_than_2_6 and pad_len > 0:
            attention_mask = torch.nn.functional.pad(attention_mask, value=0, pad=(0, pad_len))

        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Add the offsets on top (because flex interface only allows length, not start and end indices)
    mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset)

    # Finally create the block mask
    block_mask = create_block_mask(
        mask_mod=mask_function,
        B=batch_size,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=cache_position.device,
        _compile=_is_torch_greater_or_equal_than_2_6,
    )
    return block_mask


class AttentionMaskInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flash_attention_3": flash_attention_mask,
        "flex_attention": flex_attention_mask,
    }


# Global AttentionMaskInterface shared by all models which do not need to overwrite any of the existing ones
ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].
    """
    # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
    # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
    # gives exactly the sequence indices
    # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
    # cannot be part of the end of the first batch dim and the start of the 2nd one for example
    first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    # Here it would be nice to return None if we did not detect packed sequence format, i.e. if `packed_sequence_mask[:, -1] == 0`
    # but it causes issues with export
    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, BlockMask]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, BlockMask]], int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`torch.Tensor`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    # Note: it's not ideal to check the `_global_mapping` attribute instead of the object itself, however otherwise
    # full graph dynamo tracing (i.e. torch.export or compile with `fullgraph=True`) will fail on Python<3.11
    # with `torch._dynamo.exc.Unsupported: 'inline in skipfiles:Mapping.__contains__ | __contains__, skipped
    # according trace_rules.lookup SKIP_DIRS'` -- can be removed when we require Python>=3.11
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an hybrid cache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    if _is_torch_xpu_available:
        # Do not allow skip if we are compiling for decoding, but for prefill, we still allow skip to optimization the perf of 1st token generation
        allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and cache_position.shape[0] == 1)
    else:
        allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # If we detected packing format
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_bidirectional_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    encoder_hidden_states: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard bidirectional mask based on the attention implementation used (stored in the config).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is only used to infer metadata
            such as the batch size, query length, dtype, and device.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, kv_length).
            It can also be an already prepared 4D mask of shape (batch_size, 1, query_length, kv_length),
            in which case it is returned as-is.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `input_embeds` to infer the batch size, kv length and dtype.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the union of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
    """
    # Due to the logic surrounding `cache_position` in inferring query-related information, we
    # construct a dummy tensor imitating initial positions
    cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device, dtype=torch.long)

    embeds = encoder_hidden_states if encoder_hidden_states is not None else input_embeds
    # We ignore a few irrelevant arguments at the end as we do not have a (growing) cache here
    early_exit, attention_mask, _, kv_length, kv_offset = _preprocess_mask_arguments(
        config, embeds, attention_mask, cache_position, None, None, 0
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = embeds.shape[0], embeds.dtype
    mask_factory_function = bidirectional_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Allow skipping the mask creation except we have additional masking operators (and/or masks)
    allow_is_bidirectional_skip = True

    # Allow slight deviations from the base mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_bidirectional_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_bidirectional_skip = False

    # We now create the mask
    attention_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        # Additional kwargs for sdpa
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return attention_mask


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # If we detected packing format
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=sliding_window,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_chunked_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a chunked attention causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Llama4. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "chunked_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
    """
    # If we have an hybrid cache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config, or it is not set")

    # Raise if using chunked attention on context too large with FA2
    if config._attn_implementation == "flash_attention_2" and kv_length + kv_offset > chunk_size:
        raise ValueError(
            "Flash attention 2 cannot handle chunked attention, and the key-value length is larger than the chunk size so the "
            "chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    # For chunked attention and batched inputs, we need to take the number of left padding tokens into account
    # to start the chunk from the actual start of the sequence for the padded sequence
    if attention_mask is not None:
        # Only count the left padding tokens, not all of them
        left_padding_tokens = (attention_mask.cumsum(dim=-1) == torch.zeros_like(attention_mask)).sum(dim=-1)
    else:
        left_padding_tokens = torch.zeros(batch_size, device=cache_position.device, dtype=int)
    # Raise a warning for older versions if the problematic left-padding situation arises
    if (
        not _is_torch_greater_or_equal_than_2_6
        and kv_length + kv_offset > chunk_size
        and (left_padding_tokens > 0).any()
    ):
        logger.warning_once(
            "Due to limitations of your current torch version, we cannot correctly account for the left-padding "
            "when computing the chunked attention pattern. This will lead to a wrong attention mask for the padded "
            "sequences. Behavior will be undefined. Please upgrade to `torch>=2.6` to solve this issue."
        )
    mask_factory_function = chunked_causal_mask_function(chunk_size, left_padding_tokens)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    # Allow slight deviations from causal mask
    # Note that it is very important to apply this before any other deviations of the mask (such as packed sequence mask,
    # padding mask, etc) as the resulting mask may otherwise not be correct!
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # If we detected packing format
    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=chunk_size,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
}


def create_masks_for_generate(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
    **kwargs,
):
    """
    This function mimics how we create the masks in the `modeling_xxx.py` files, and is used in places like `generate`
    in order to easily create the masks in advance, when we compile the forwards with Static caches.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # The attribute reside in the text config for composite models
    effective_config = config.get_text_config()
    # Prepare the mask args
    mask_kwargs = {
        "config": effective_config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
    }

    # If the attribute exist, we need several masks
    if hasattr(effective_config, "layer_types"):
        causal_masks = {}
        for layer_pattern in set(effective_config.layer_types):
            causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
        return causal_masks
    # In this case, all layers are sliding
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    # In this case, all layers are chunked
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    # All layers use standard causal attention
    return create_causal_mask(**mask_kwargs)


# Below are utilities to pretty-print the different masks
# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"
GREY_SQUARE = "∙"
LOW_TRIANGLE = "⬕"
UPPER_TRIANGLE = "⬔"


def get_style(style):
    if style == "majong":
        BLACK_SQUARE = "🀞"  # Full block (represents "on" or active)
        BLACK_SQUARE = "🀙"  # Full block (represents "on" or active)
        WHITE_SQUARE = "🀆"  # "▒"  # Light shade (represents "off" or inactive)
        LOW_TRIANGLE = "🀛"  # Lower left triangle (stylized indication)
        UPPER_TRIANGLE = "🀛"  # Upper left triangle (stylized indication)
    else:
        BLACK_SQUARE = "█"  # Full block (represents "on" or active)
        WHITE_SQUARE = "░"  # "▒"  # Light shade (represents "off" or inactive)
        LOW_TRIANGLE = "▙"  # Lower left triangle (stylized indication))
        UPPER_TRIANGLE = "▜"  # Upper left triangle (stylized indication)

    return BLACK_SQUARE, WHITE_SQUARE, LOW_TRIANGLE, UPPER_TRIANGLE


# LOW_TRIANGLE = UPPER_TRIANGLE = "⟍"   # Upper right triangle (stylized indication)

YELLOW_SQUARE = f"{YELLOW}{BLACK_SQUARE}{RESET}"
GREEN_SQUARE = f"{GREEN}{BLACK_SQUARE}{RESET}"


def tensor_to_mask_visual(original_tensor: torch.Tensor, grid_size=(20, 40), style="majong") -> str:
    BLACK_SQUARE, WHITE_SQUARE, LOW_TRIANGLE, UPPER_TRIANGLE = get_style(style)
    h, w = original_tensor.shape
    max_h, max_w = grid_size
    if not (h < max_h and w < max_w):
        # Preserve aspect ratio within max grid size
        aspect_ratio = 2 * w / h
        if aspect_ratio > 1:
            w = max_w
            h = min(max_h, max(1, round(max_w / aspect_ratio)))
        else:
            h = max_h
            w = max(1, round(max_h * aspect_ratio))

        # Step 1: Rescale tensor by average pooling
        tensor = original_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        tensor = F.adaptive_avg_pool2d(tensor, output_size=(h, w))[0, 0]  # Remove extra dims
    else:
        tensor = original_tensor

    # Step 3: Build the string representation
    result = []
    for i in range(h):
        row = ""
        for j in range(w):
            if tensor[i, j] == 1:
                row += BLACK_SQUARE
            elif tensor[i, j] == 0:
                row += WHITE_SQUARE
            else:
                if j > 0:
                    if tensor[i, j - 1] == 1:
                        row += LOW_TRIANGLE
                    elif tensor[i, j - 1] == 0:
                        row += UPPER_TRIANGLE
                    else:
                        row += BLACK_SQUARE if tensor[i, j] == 1 else WHITE_SQUARE
                else:
                    row += (
                        BLACK_SQUARE
                        if tensor[i, j] == 1
                        else (
                            WHITE_SQUARE
                            if tensor[i, j] == 0
                            else (UPPER_TRIANGLE if tensor[i, j + 1] == 1 else LOW_TRIANGLE)
                        )
                    )
        result.append(row)

    return "\n".join(result)


class AttentionMask(torch.Tensor):
    def __new__(cls, data, style=None):
        # Create a new instance of AttentionMask as a Tensor
        cls.style = style
        return torch.Tensor._make_subclass(cls, data, require_grad=False)

    def __init__(self, data):
        # You can initialize any additional metadata here if needed
        pass

    def to_string(self, grid_size=(20, 40), limit=4):
        """Returns a string representation of the block mask."""
        dense_mask = self
        *batch_dims, num_rows, num_cols = dense_mask.shape
        total_vis = []

        for idx, batch_idx in enumerate(itertools.product(*[range(i) for i in batch_dims])):
            if idx == limit:
                total_vis.append("...")
                total_vis.append("To print out more, set AttentionMask.to_string(limit=N)")
                total_vis.append("You can also index (AttentionMask[batch, head]) to choose a specific batch or head")
                break
            block_vis = tensor_to_mask_visual(dense_mask[batch_idx], grid_size=grid_size, style=self.style)
            total_vis.append(block_vis)

        total_vis.append(f"torch.Tensor(shape={tuple(self.shape)}, dtype={self.dtype})")
        return "\n".join(total_vis)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, style: Optional[str] = None) -> "AttentionMask":
        res = cls(tensor)
        res.style = style
        return res
# ==================== inline: models/llada2_vl/configuration_llada2_vl.py ====================
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/lladamoe/modular_lladamoe.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_lladamoe.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


ALLOWED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "linear_attention",  # used in minimax
)


def layer_type_validation(layer_types, num_hidden_layers=None):
    """Check that `layer_types` is correctly defined."""
    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES}")
    if num_hidden_layers is not None and num_hidden_layers != len(layer_types):
        raise ValueError(
            f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
            f"({len(layer_types)})"
        )

class LLaDAMoEVisionConfig(PretrainedConfig):
    # model_type = "lladamoe"
    model_type = "lladavision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class LLaDA2MoeConfig(PretrainedConfig):
    model_type = "llada2_vl"

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,  # llada2 only
        use_qk_norm=False,
        use_bias=True,  # llada2 only
        rms_norm_eps=1e-05,
        norm_head=False,  # llada2 only
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor

        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)


class LLaDA2VLMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LLaDAMoEModel`]. It is used to instantiate a
    LLaDAMoE model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `LLaDAMoETextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `LLaDAMoEVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.

    ```python
    >>> from transformers import LLaDAMoEForConditionalGeneration, LLaDAMoEConfig

    >>> # Initializing a LLaDAMoE style configuration
    >>> configuration = LLaDAMoEConfig()

    >>> # Initializing a model from the LLaDAMoE style configuration
    >>> model = LLaDAMoEForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # model_type = "lladamoe"
    model_type = "llada2vl"
    sub_configs = {"vision_config": LLaDAMoEVisionConfig, "text_config": LLaDA2MoeConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):  
        self.is_composition = True
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are same as for `LLaDAMoETextConfig`
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        
        # Attention implementation to use. It sets it recursively on sub-configs so we call it again in the end
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


__all__ = ["LLaDA2VLMoEConfig", "LLaDA2MoeConfig", "LLaDAMoEVisionConfig"]
# ==================== inline: models/llada2_vl/modeling_llada2_vl_moe_bd_sdpa_hf.py (intra-imports stripped) ====================
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/lladamoe/modular_lladamoe.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_lladamoe.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
# from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
# from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
# from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
# from models.llada2_vl.configuration_lladamoe import LLaDAMoEConfig, LLaDAMoETextConfig, LLaDAMoEVisionConfig
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    is_flash_attn_2_available,
    logging,
    # TransformersKwargs,
    # auto_docstring, 
    # can_return_tuple, 
    logging
)
import torch.distributed as dist
from torch.distributed._tensor import Shard
# if is_flash_attn_2_available():
#     from transformers.modeling_flash_attention_utils import _flash_attention_forward

# if is_flash_attn_2_available():
# from flash_attn import flash_attn_varlen_func
# from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from flash_attn import flash_attn_func, flash_attn_varlen_func

from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

# if is_liger_kernel_available():
#     from liger_kernel.ops.swiglu import LigerSiLUMulFunction
#     from liger_kernel.transformers.rms_norm import LigerRMSNorm
#     from liger_kernel.transformers.rope import liger_rotary_pos_emb

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

# from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.layers.rotary import apply_rotary_emb

import numpy as np


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        # ulysses sp patch: qkv projection
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)

        unpadded_dim_size = cu_seqlens[-1]
        # if get_parallel_state().ulysses_enabled:
        #     qkv = gather_seq_scatter_heads(qkv, seq_dim=1, head_dim=2)
        #     sp_padding_size = qkv.size(1) - unpadded_dim_size
        #     if sp_padding_size > 0:
        #         qkv = unpad_tensor(qkv, dim=1, padding_size=sp_padding_size)
        q, k, v = qkv.unbind(0)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)

        # ulysses sp patch: o projection
        # if get_parallel_state().ulysses_enabled:
        #     attn_output = pad_tensor(attn_output, dim=0, padding_size=sp_padding_size)
        #     attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)

        return attn_output


QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "flash_attention_2": Qwen2_5_VLVisionFlashAttention2,
    "sdpa": Qwen2_5_VLVisionSdpaAttention,
}

def _unpack_router_logits(router_outputs):
    """
    Unpack the router tuple for balance loss calculation.
    """
    total_router_logits = []
    total_expert_indexes = []
    for router_output in router_outputs:
        if router_output[0] is not None:
            router_logits, expert_indexes = router_output
            total_router_logits.append(router_logits.unsqueeze(0))
            total_expert_indexes.append(expert_indexes.unsqueeze(0))
    return torch.cat(total_router_logits, dim=0), total_expert_indexes


def router_z_loss_func(router_logits: torch.Tensor, labels: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [num_layers, batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_layers, num_groups, tokens_per_group, _ = router_logits.shape

    labels_mask = (labels[None, ..., None].expand_as(router_logits) != -100).long()

    ori_dtype = router_logits.dtype
    if ori_dtype == torch.bfloat16:
        loss_func_inputs = (router_logits * labels_mask).to(torch.float32)
    else:
        loss_func_inputs = router_logits * labels_mask
    log_z = torch.logsumexp(loss_func_inputs, dim=-1).to(ori_dtype)
    z_loss = log_z**2

    return torch.sum(z_loss) / (num_layers * num_groups * tokens_per_group)


# checked same as Qwen2.5-VL
class LLaDAMoE_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

# checked same as Qwen2.5-VL
class LLaDAMoE_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

# checked same as Qwen2.5-VL
class LLaDAMoE_VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

# checked same as Qwen2.5-VL & LLaDAMoE
class LLaDA2MoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LLaDA2MoERMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LLaDA2MoERMSNorm)

# checked same as Qwen2.5-VL
class LLaDAMoE_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LLaDA2MoERMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

# checked same as Qwen2.5-VL
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# checked same as Qwen2.5-VL
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

# checked same as Qwen2.5-VL
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# checked same as Qwen2.5-VL
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# checked same as Qwen2.5-VL
class LLaDAMoE_VLVisionAttention(nn.Module):
    def __init__(self, config: LLaDAMoEVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

# checked same as Qwen2.5-VL
# class LLaDAMoE_VLVisionBlock(GradientCheckpointingLayer):
class LLaDAMoE_VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LLaDA2MoERMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = LLaDA2MoERMSNorm(config.hidden_size, eps=1e-6)
        # self.attn = LLaDAMoE_VLVisionAttention(config=config)
        attn_implementation = "flash_attention_2"
        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = LLaDAMoE_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states



# from .modeling_llada2_moe import LLaDA2MoePreTrainedModel
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    # config_class = LLaDA2MoeConfig
    config_class = LLaDA2VLMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    # _no_split_modules = ["LLaDAMoE_VLDecoderLayer", "LLaDAMoE_VLVisionBlock"]
    _no_split_modules = ["LLaDA2MoeDecoderLayer", "LLaDAMoE_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    _supports_flash_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# checked same as Qwen2.5-VL
class LLaDAMoE_VisionTransformerPretrainedModel(LLaDA2MoePreTrainedModel):
    config: LLaDAMoEVisionConfig
    _no_split_modules = ["LLaDAMoE_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = LLaDAMoE_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = LLaDAMoE_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([LLaDAMoE_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = LLaDAMoE_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False


    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens


    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # sp patch: use all-to-all to get full sequence of hidden_states for window attention
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        unpadded_dim_size = cu_seqlens[-1]
        # if get_parallel_state().sp_enabled:
        #     hidden_states = gather_seq_scatter_heads(
        #         hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        #     )
        #     sp_padding_size = hidden_states.size(0) - unpadded_dim_size
        #     if sp_padding_size > 0:
        #         hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # sp patch: use all-to-all to reshape back to sequence-splitting hidden_states for window attention
        # if get_parallel_state().sp_enabled:
        #     if sp_padding_size > 0:
        #         hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
        #     hidden_states = gather_heads_scatter_seq(
        #         hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        #     )

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)

        # sp patch: use all-to-all to get full sequence of hidden_states for window attention
        # if get_parallel_state().sp_enabled:
        #     sp_padding_size = hidden_states.size(0) - unpadded_dim_size
        #     hidden_states = gather_seq_scatter_heads(
        #         hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        #     )
        #     if sp_padding_size > 0:
        #         hidden_states = unpad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)

        hidden_states = hidden_states[reverse_indices, :]

        # sp patch: use all-to-all to reshape back to sequence-splitting hidden_states for window attention
        # if get_parallel_state().sp_enabled:
        #     # print(f"[Rank{torch.distributed.get_rank()}] hidden_states4 (before pad): {hidden_states.shape}", flush=True)
        #     if sp_padding_size > 0:
        #         hidden_states = pad_tensor(hidden_states, dim=0, padding_size=sp_padding_size)
        #     hidden_states = gather_heads_scatter_seq(
        #         hidden_states, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
        #     )

        return hidden_states

    def dummy_forward(self):
        if getattr(self, "_dummy_data", None) is None:
            pixel_values = torch.randn((4, 3 * 2 * 14 * 14), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32, device=self.device)
            self._dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        return self(**self._dummy_data)

# checked same as Qwen2.5-VL
# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Llava outputs, with hidden states and attentions.
#     """
# )
@dataclass
class LLaDA2MoE_VLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    all_router_tuple: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class LLaDA2MoE_DecoderOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """
    ## todo
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    all_router_tuple: Optional[tuple[torch.FloatTensor]] = None


from types import SimpleNamespace
from functools import partial

# customized functions
def get_position_id(main_func, self, **kwargs):
    # must be a global func for multiproceesing serialize
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, bs, l)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


# class LLaDA2MoeRotaryEmbedding(nn.Module):
#     def __init__(self, config: LLaDA2MoeConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         # inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
#         # position_ids_expanded = position_ids[:, None, :].float()
#         inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
#         position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos() * self.attention_scaling
#             sin = emb.sin() * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # if is_liger_kernel_available():
        #     return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
        # else:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def update_expert_bias(self, selected_experts, total_tokens):
        with torch.no_grad():
            # 统计专家选择次数
            selected_flat = selected_experts.view(-1)
            expert_counts = torch.bincount(
                selected_flat,
                minlength=self.num_experts
            ).float().to(self.expert_bias.device)
            
            # 计算负载误差
            target = (total_tokens * self.top_k) / self.num_experts
            load_error = target - expert_counts
            
            # RMS归一化
            eps = 1e-12
            rms = torch.sqrt(torch.mean(load_error**2) + eps)
            normalized_error = load_error / rms
            
            # 更新偏置
            aux_loss_coef = 1e-3
            self.expert_bias.data += aux_loss_coef * normalized_error

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        # scores_for_routing = scores
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        if self.training:
            total_tokens = batch_size * sequence_length
            self.update_expert_bias(topk_idx, total_tokens)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class LLaDA2MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_size),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None):
        
        gate_proj_out = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
        up_proj_out = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))

        out = self.act_fn(gate_proj_out) * up_proj_out
        out = torch.matmul(out, self.down_proj[expert_idx].transpose(0, 1))
       
        return out

    def reset_parameters(self):
        """
        Initialize the parameters of all expert networks.
        Uses different initialization strategies for different projection layers.
        """
        for expert_id in range(self.num_experts):
            nn.init.kaiming_uniform_(self.gate_proj[expert_id], a=math.sqrt(5))            
            nn.init.kaiming_uniform_(self.up_proj[expert_id], a=math.sqrt(5))
            nn.init.xavier_uniform_(self.down_proj[expert_id])



class LLaDA2MoeSparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        print(f"Load model in {self.config.model_type} mode!")
        self._setup_experts()

        self.gate = LLaDA2MoeGate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = LLaDA2MoeMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )

        # self.register_buffer("expert_bias", torch.zeros((config.num_experts)))
    
    def _setup_fuse_moe_experts(self):
        self.experts = LLaDA2MoeExperts(self.config)

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                LLaDA2MoeMLP(config=self.config, intermediate_size=self.config.moe_intermediate_size)
                for _ in range(self.config.num_experts)
            ]
        )

    def _fuse_moe_forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)



        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.experts(
            hidden_states, routing_weights=topk_weight, selected_experts=topk_idx
        ).reshape(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    def _forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    def forward(self, hidden_states):
        return self._forward(hidden_states)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# # # Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
# class LLaDA2MoeAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         if layer_idx is None:
#             logger.warning_once(
#                 f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
#                 "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
#                 "when creating this class."
#             )

#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = config.head_dim or self.hidden_size // self.num_heads
#         partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
#         self.rope_dim = int(self.head_dim * partial_rotary_factor)
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = False
        

#         self.query_key_value = nn.Linear(
#             self.hidden_size,
#             (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
#             bias=config.use_qkv_bias,
#         )

#         self.query_layernorm = LLaDA2MoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         self.key_layernorm = LLaDA2MoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
#         self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

#         self.rope_scaling = config.rope_scaling

#     def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if "padding_mask" in kwargs:
#             warnings.warn(
#                 "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
#             )

#         bsz, q_len, _ = hidden_states.size()

#         qkv = self.query_key_value(hidden_states)
#         qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

#         query_states, key_states, value_states = qkv.split(
#             [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
#         )
#         query_states = query_states.transpose(1, 2)
#         key_states = key_states.transpose(1, 2)
#         value_states = value_states.transpose(1, 2)

#         query_states = self.query_layernorm(query_states)
#         key_states = self.key_layernorm(key_states)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             if self.layer_idx is None:
#                 raise ValueError(
#                     f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
#                     "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
#                     "with a layer index."
#                 )
#             kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#         # query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])


#         if past_key_value is not None:
#             cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {attn_weights.size()}"
#             )
#         # attention_mask = None
#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights + attention_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, -1)

#         attn_output = self.dense(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        self.query_layernorm = LLaDA2MoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # attention_mask = None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LLaDA2MoeFlexAttention(LLaDA2MoeAttention):
    # Adapted from LLaDA2MoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LLaDA2MoeModel is using LLaDA2MoeFlexAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(
            bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # print(attention_mask)

        

        attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["flex_attention"](
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )
        # attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, None, past_key_value


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )



# from flash_attn_interface import flash_attn_func

# # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->LLaDA2Moe
class LLaDA2MoeFlashAttention2(LLaDA2MoeAttention):
    """
    LLaDA2Moe flash attention module. This module inherits from `LLaDA2MoeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LLaDA2MoeFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling["mrope_section"])


        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently cast in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slow down training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LLaDA2MoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, position_ids=position_ids
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _prepare_fa2_from_position_ids(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
    ):
        # query = query.reshape(-1, query.size(-2), query.size(-1))
        # key = key.reshape(-1, key.size(-2), key.size(-1))
        # value = value.reshape(-1, value.size(-2), value.size(-1))

        query = query.view(-1, query.size(-2), query.size(-1))
        key = key.view(-1, key.size(-2), key.size(-1))
        value = value.view(-1, value.size(-2), value.size(-1))
        position_ids = position_ids.flatten()
        indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
        cu_seqlens = torch.cat(
            (
                indices_q[position_ids == 0],
                torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
            )
        )
        max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length
        return (query, key, value, indices_q, (cu_seqlens, cu_seqlens), (max_length, max_length))

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, position_ids=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            query_length (`int`):
                The length of the query sequence in terms of tokens. This represents the number of tokens in the
                `query_states` tensor along the sequence dimension. It is used to determine the effective sequence
                length for attention computations.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LLaDA2MoeFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1


        # print(f'rank: {dist.get_rank()}, position_ids: {position_ids.size()}, indices_q: {indices_q.shape}')

        # attention_mask = None
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            # batch_size = query_states.shape[0]
            # query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            #     query_states, key_states, value_states, attention_mask, query_length
            # )

            # query_states, key_states, value_states, _, cu_seq_lens, max_seq_lens = self._prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids[0])

            # cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            batch_size, seq_len = attention_mask.shape
            query_states, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(query_states, attention_mask)
            key_states, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(key_states, attention_mask)
            value_states, _, _, _ , _= unpad_input(value_states, attention_mask)

            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seqlen_q, max_seqlen_k
                
            # attn_output_unpad
            # attn_output = flash_attn_varlen_func(
            #     query_states,
            #     key_states,
            #     value_states,
            #     cu_seqlens_q=cu_seqlens_q,
            #     cu_seqlens_k=cu_seqlens_k,
            #     max_seqlen_q=max_seqlen_in_batch_q,
            #     max_seqlen_k=max_seqlen_in_batch_k,
            #     softmax_scale=softmax_scale,
            #     causal=causal,
            # )

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
                            # dropout_p=dropout,

            attn_output = pad_input(attn_output, indices_q, batch_size, query_length)
            # attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->LLaDA2Moe
class LLaDA2MoeSdpaAttention(LLaDA2MoeAttention):
    """
    LLaDA2Moe attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LLaDA2MoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LLaDA2MoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LLaDA2MoeModel is using LLaDA2MoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # query_states, key_states = apply_multimodal_rotary_pos_emb(
        #     query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        # )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attention_mask = None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value

'''
ATTENTION_CLASSES = {
    "eager": LLaDA2MoeAttention,
    "flash_attention_2": LLaDA2MoeFlashAttention2,
    "sdpa": LLaDA2MoeSdpaAttention,
}
'''



ATTENTION_CLASSES = {
    "eager": LLaDA2MoeSdpaAttention,
    "flash_attention_2": LLaDA2MoeSdpaAttention,
    "sdpa": LLaDA2MoeSdpaAttention,
}



'''
ATTENTION_CLASSES = {
    "eager": LLaDA2MoeFlexAttention,
    "flash_attention_2": LLaDA2MoeFlexAttention,
    "sdpa": LLaDA2MoeFlexAttention,
}
'''

'''
ATTENTION_CLASSES = {
    "eager": LLaDA2MoeFlashAttention2,
    "flash_attention_2": LLaDA2MoeFlashAttention2,
    "sdpa": LLaDA2MoeFlashAttention2,
}
'''




class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = LLaDA2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

def calculate_pack_position_ids(
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_values_length: int = 0,
    cu_lengths_list: Optional[List[torch.Tensor]] = None,
):
    """
    计算 position_ids。

    如果提供了 cu_lengths_list (累积序列长度列表)，则为打包（packed）序列生成 position_ids，
    每个子序列的 position_ids 从 0 开始。
    cu_lengths_list 的格式为 [tensor([0, len1, len1+len2, ...]), tensor([0, lenA, lenA+lenB, ...])].

    否则，生成标准的连续 position_ids。
    """
    if position_ids is not None:
        # 如果已经提供了 position_ids，直接返回
        return position_ids

    # 确定 device 和 batch_size, seq_length
    if input_ids is not None:
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        device = inputs_embeds.device
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    # ----- 新增的核心逻辑：处理打包序列 (累积长度格式) -----
    if cu_lengths_list is not None:
        # 用于存储每个样本最终的 position_ids 张量
        all_position_ids = []
        
        for i in range(batch_size):
            # 获取当前样本的累积长度张量，例如 tensor([0, 420, 840, ...])
            cu_seqlens = cu_lengths_list[i].to(device)
            
            # 1. 计算每个子序列的起始位置
            # 例如, cu_seqlens = [0, 3, 8, 10] -> starts = [0, 3, 8]
            starts = cu_seqlens[:-1]
            
            # 2. 计算每个子序列的实际长度
            # 例如, cu_seqlens = [0, 3, 8, 10] -> lengths = [3, 5, 2]
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]

            # 3. 创建一个从0开始的全局位置id，总长度为打包序列的总长
            # 例如, 总长为10, 全局id为 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            total_len = cu_seqlens[-1].item()
            global_positions = torch.arange(total_len, device=device, dtype=torch.long)

            # 4. 创建一个“减法掩码”，将每个子序列的起始位置广播到其对应长度
            # 例如, starts=[0, 3, 8], lengths=[3, 5, 2]
            # -> subtraction_mask = [0,0,0, 3,3,3,3,3, 8,8]
            subtraction_mask = torch.repeat_interleave(starts, lengths)
            
            # 5. 从全局位置中减去起始位置，得到局部位置
            # [0,1,2, 3,4,5,6,7, 8,9] - [0,0,0, 3,3,3,3,3, 8,8]
            # -> [0,1,2, 0,1,2,3,4, 0,1]
            current_pos_ids = global_positions - subtraction_mask
            
            all_position_ids.append(current_pos_ids)
        
        # 由于每个打包后的样本总长度可能不同，我们需要将它们填充到同一长度
        position_ids = torch.nn.utils.rnn.pad_sequence(
            all_position_ids, 
            batch_first=True, 
            padding_value=0 # padding的值通常不重要，因为会被attention_mask忽略
        )
        
        # 确保填充后的长度与输入的 seq_length 匹配
        if position_ids.shape[1] < seq_length:
            pad_right = seq_length - position_ids.shape[1]
            position_ids = torch.nn.functional.pad(position_ids, (0, pad_right), 'constant', 0)
    else:
        # ----- 原始逻辑：处理非打包序列 -----
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    return position_ids



class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaDA2MoeDecoderLayer`]
    Args:
        config: LLaDA2MoeConfig
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LLaDA2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LLaDA2MoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    # @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cu_lengths_list: Optional[List] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False


        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            '''
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            '''
            position_ids = calculate_pack_position_ids(input_ids, inputs_embeds, position_ids, past_key_values_length, cu_lengths_list)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # # TODO flash attention 2 can not support custom attention mask
        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # if self._use_sdpa and not output_attentions:
        #     # output_attentions=True can not be supported when using SDPA, and we fall back on
        #     # the manual implementation that requires a 4D causal mask in all cases.
        #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #     )
        if hasattr(attention_mask, 'dim') and attention_mask.dim() == 2:
            if self._use_sdpa and not output_attentions:
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                # attention_mask = _prepare_4d_causal_attention_mask(
                #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                # )
                # att1 = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
                if attention_mask is not None:
                    attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
                else:
                    attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                    )

        causal_mask = None
        # attention_mask = None

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )



# from .modeling_llada2_moe import LLaDA2MoeModel

# @auto_docstring
class LLaDA2MoE_VLModel(LLaDA2MoePreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    # config: LLaDAMoEConfig
    config: LLaDA2VLMoEConfig
    # _no_split_modules = ["LLaDAMoE_VLDecoderLayer", "LLaDAMoE_VLVisionBlock"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.text_config._attn_implementation = "flash_attention_2"
        # config.text_config._attn_implementation = "eager"
        self.visual = LLaDAMoE_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = LLaDA2MoeModel._from_config(config.text_config)

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # import os
        # import json
        # from safetensors.torch import load_file

        # index_path = os.path.join(self.lladamoe_path, "model.safetensors.index.json")
        # with open(index_path, "r") as f:
        #     index = json.load(f)

        # tensor_name = "lm_head.weight"

        # shard_filename = index["weight_map"].get(tensor_name)

        # shard_path = os.path.join(self.lladamoe_path, shard_filename)
        # all_tensors_in_shard = load_file(shard_path, device="cpu")

        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # lm_head_weights = all_tensors_in_shard[tensor_name]
        # with torch.no_grad():
        #     self.lm_head.weight.copy_(lm_head_weights)

        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None and hasattr(attention_mask, 'dim') and attention_mask.dim() == 2:
                attention_mask = attention_mask == 1
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None and hasattr(attention_mask, 'dim') and attention_mask.dim() == 2:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                if attention_mask is not None and hasattr(attention_mask, 'dim') and attention_mask.dim() == 2:
                    position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
                else:
                    position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None and hasattr(attention_mask, 'dim') and attention_mask.dim() == 2:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        cu_lengths_list: Optional[List] = None,
        **kwargs,
        # **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LLaDA2MoE_VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        # attention_mask = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if position_ids is None:
        #     if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
        #         position_ids, rope_deltas = self.get_rope_index(
        #             input_ids,
        #             image_grid_thw,
        #             video_grid_thw,
        #             second_per_grid_ts=second_per_grid_ts,
        #             attention_mask=attention_mask,
        #         )
        #         self.rope_deltas = rope_deltas
        #     else:
        #         batch_size, seq_length, _ = inputs_embeds.shape
        #         position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        #         position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        #         if cache_position is not None:
        #             delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
        #         else:
        #             delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
        #         delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
        #         position_ids = position_ids + delta.to(position_ids.device)

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device

        #     if replace_position is not None:
        #         position_ids = torch.arange(replace_position[0], replace_position[1], device=device, dtype=torch.long).unsqueeze(0)
        #     else:
        #         position_ids = torch.arange(inputs_embeds.shape[1], device=device, dtype=torch.long).unsqueeze(0)

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        #     )
        #     position_ids = position_ids.unsqueeze(0)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            cu_lengths_list=cu_lengths_list,
            **kwargs,
        )

        output = LLaDA2MoE_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_router_tuple=outputs.router_logits,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for LLaDAMoE_VL causal language model (or autoregressive) outputs.
#     """
# )
@dataclass
class LLaDAMoE_VLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    z_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def get_transfer_index(logits, mask_index, x, block_end, num_transfer_tokens, temperature, top_p, top_k, remasking, minimal_topk=1, 
                        threshold=None, opt_softmax=False):
    # token
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)  # b, l
    if top_p is not None and top_p < 1:
        logits_with_noise = top_p_logits(logits_with_noise, top_p)
    if top_k is not None:
        logits_with_noise = top_k_logits(logits_with_noise, top_k)

    x0 = torch.argmax(logits_with_noise, dim=-1) 

    # index
    if remasking in ['low_confidence', 'random']:
        if opt_softmax:
            if remasking == 'low_confidence':
                p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(logits.dtype)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0[mask_index], -1)), -1)  # b, l
                confidence = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
                confidence[mask_index] = x0_p
                # confidence = torch.where(mask_index, x0_p, -np.inf)
                confidence[:, block_end:] = -np.inf

            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
            else:
                raise NotImplementedError(remasking)
        else:
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)    
    
    elif remasking in ['neg_entropy', 'top_k_margin']:
        certainty_scores = torch.full_like(x0, -np.inf, dtype=logits.dtype)
        
        if opt_softmax:
            masked_logits = logits[mask_index]
            if masked_logits.numel() == 0:
                pass
            else:
                p = F.softmax(masked_logits.to(torch.float32), dim=-1).to(logits.dtype)
                
                if remasking == 'neg_entropy':
                    epsilon = 1e-10
                    log_probs = torch.log(p + epsilon)
                    scores = -torch.sum(p * log_probs, dim=-1)
                else:  # 'top_k_margin'
                    if p.shape[-1] < 2:
                        scores = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
                    else:
                        sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                        scores = sorted_probs[..., 0] - sorted_probs[..., 1]
            
                certainty_scores[mask_index] = scores
        else:
            p = F.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)
            
            if remasking == 'neg_entropy':
                epsilon = 1e-10
                log_probs = torch.log(p + epsilon)
                scores = -torch.sum(p * log_probs, dim=-1) 
            else:  # 'top_k_margin'
                if p.shape[-1] < 2:
                    scores = torch.zeros_like(p[..., 0])
                else:
                    sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                    scores = sorted_probs[..., 0] - sorted_probs[..., 1]
            certainty_scores = torch.where(mask_index, scores, -np.inf)

        confidence = certainty_scores
        if block_end is not None:
            confidence[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        
    else:
        raise NotImplementedError(f"Remasking strategy '{remasking}' is not implemented.")
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

class LLaDA2MoE_VLForConditionalGeneration(LLaDA2MoePreTrainedModel, GenerationMixin):
    # _checkpoint_conversion_mapping = {
    #     "^visual": "model.visual",
    #     r"^model(?!\.(language_model|visual))": "model.language_model",
    # }
    # _tied_weights_keys = ["model.lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config, args=None, tokenizer=None, image_processor=None):
        super().__init__(config)
        self.model = LLaDA2MoE_VLModel(config)
        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        self.tokenizer=tokenizer
        self.image_processor = image_processor
        self.num_embeddings = self.model.language_model.get_input_embeddings().num_embeddings

        # try:
        #     self.model.resize_token_embeddings(
        #         self.num_embeddings + 4
        #     )
        # except:
        #     self.model.resize_token_embeddings(
        #         self.num_embeddings + 4, mean_resizing=False
        #     )
        
        # self.tokenizer.add_special_tokens(
        #     {
        #         "additional_special_tokens": [
        #             f"<pad_token_{i}>"
        #             for i in range(self.num_embeddings - len(self.tokenizer))
        #         ]
        #     }
        # )
        # self.tokenizer.add_special_tokens(
        #     {
        #         "additional_special_tokens": ["<image>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        #     }
        # )
        # self.img_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        # self.img_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        # self.img_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        # self.img_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self.img_token_id = 157184
        self.img_start_id = 157185
        self.img_end_id = 157186
        self.img_pad_id = 157187


        # if args.data.datasets_type == "llada2_vl_pretrain_proj":
        #     self.modules_to_freeze = ['']
        #     self.modules_to_unfreeze = ['model.visual.merger']
        #     self.model.requires_grad_(False)
        # else:
        #     self.modules_to_freeze = ['']
        #     self.modules_to_unfreeze = ['']
        #     self.model.requires_grad_(True)


        self.modules_to_freeze = ['']
        self.modules_to_unfreeze = ['']
        self.model.requires_grad_(True)

        # self.modules_to_freeze = ['']
        # self.model.requires_grad_(False)
        # self.modules_to_unfreeze = ['model.visual.merger']

        # for module_name in self.modules_to_freeze:
        #     if "." in module_name:
        #         module = self
        #         for sub_module_name in module_name.split("."):
        #             module = getattr(module, sub_module_name, None)
        #             if module is None:
        #                 break
        #         else:
        #             module.requires_grad_(False)
        #     else:
        #         module = getattr(self, module_name, None)
        #         if module is not None:
        #             module.requires_grad_(False)

        for module_name in self.modules_to_unfreeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(True)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(True)

        # para = sum(p.numel() for p in self.parameters())
        # para_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    # def get_output_embeddings(self):
    #     return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def get_position_id_func(self):  # used in data_transform during training
        fake_model = SimpleNamespace(config=self.config)
        return partial(get_position_id, LLaDA2MoE_VLModel.get_rope_index, fake_model)


    # @can_return_tuple
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        cu_lengths_list: Optional[List] = None,
        z_loss: Optional[bool] = True,
        **kwargs,
        # **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LLaDAMoE_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LLaDAMoE_VLForConditionalGeneration

        >>> model = LLaDAMoE_VLForConditionalGeneration.from_pretrained("LLaDAMoE/LLaDAMoE-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("LLaDAMoE/LLaDAMoE-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        # attention_mask = None

        output_router_logits = True

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # image_mask = kwargs["image_mask"] if "image_mask" in kwargs else None
        image_mask = input_ids == self.img_pad_id

        if inputs_embeds is None:
            # inputs_embeds = self.model.embed_tokens(input_ids)

            # inputs_embeds = self.model.language_model.embed_tokens(input_ids)
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            

            # if self.training and get_parallel_state().sp_enabled:
            #     inputs_embeds = gather_seq_scatter_heads(
            #         inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            #     )

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                # if self.training and get_parallel_state().sp_enabled:
                #     image_embeds = gather_seq_scatter_heads(
                #         image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                #     )

                n_image_tokens = image_mask.sum().long().item()

                image_embeds = image_embeds[:n_image_tokens]
                n_image_features = image_embeds.shape[0]

                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            # if self.training and get_parallel_state().sp_enabled:
            #     inputs_embeds = gather_heads_scatter_seq(
            #         inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            #     )

            # get_parallel_state().fsdp_enabled pixel_values is None
            # if self.training and get_parallel_state().fsdp_enabled and pixel_values is None:
            #     fake_embeds = self.model.visual.dummy_forward().mean() * 0.0
            #     inputs_embeds = inputs_embeds + fake_embeds
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            cu_lengths_list=cu_lengths_list,
            **kwargs,
        )


        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model.lm_head(hidden_states[:, slice_indices, :])

        loss = None

        if z_loss:
            router_tuple = outputs.all_router_tuple
            router_logits, top1_expert_index = _unpack_router_logits(router_tuple)
            top1_expert_index = torch.cat(top1_expert_index, dim=0)
            router_logits = router_logits.view(router_logits.shape[0], len(labels), -1, router_logits.shape[-1])
            z_loss = router_z_loss_func(router_logits, labels)
        else:
            z_loss = 0

        # loss = loss + 0.001 * z_loss
        # loss = loss + 4.375e-06 * z_loss
        # z_loss = None


        return LLaDAMoE_VLCausalLMOutputWithPast(
            loss=loss,
            z_loss=z_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # LLaDAMoE-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do assisted decoding
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                batch_size, seq_length = model_inputs["position_ids"].shape
                device = model_inputs["position_ids"].device
                position_ids = torch.arange(seq_length, device=device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     temperature: int = 0.0,
    #     block_length: int = 32,
    #     steps: int = 32,
    #     gen_length: int = 2048,
    #     top_p: Optional[int] = None,
    #     top_k: Optional[int] = None,
    #     eos_early_stop: bool = False,
    #     minimal_topk: int = 1,
    #     threshold: float = 0.95,
    #     eos_id: int = 156892,
    #     mask_id: int = 156895,
    # ):
    @torch.no_grad()
    def generate_dllm(self, data, steps=128, gen_length=128, block_length=32, temperature=0.0, top_p=None,top_k=None,
                cfg_scale=0., remasking='low_confidence',threshold=0.95,minimal_topk=1, 
                eos_early_stop=True, mask_id=156895,eos_id=156892,opt_softmax=True
                ):
        prompt = data['input_ids']
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        # data['input_ids'] = x
        # data["attention_mask"] = x.ne(self.tokenizer.pad_token_id)
        # data["attention_mask"] = None


        prompt_index = (x != mask_id)
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = prompt.shape[1] + (num_block+1) * block_length
            block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            i = 0
            while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:               

                mask_index = (x == mask_id)
                mask_index[:, current_block_end:] = False
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    data['input_ids'] = x_
                    logits = self(**data, z_loss=False).logits
                    # logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    data['input_ids'] = x
                    logits = self(**data, z_loss=False).logits

                logits_shifted = logits[:, :-1, :]
                mask_index_shifted = mask_index[:, 1:]
                x_shifted = x[:, 1:]
                block_end_shifted = max(0, current_block_end - 1)

                # print(f'rank: {dist.get_rank()}, i: {i}')
                x0_shifted, transfer_index_shifted = get_transfer_index(
                logits_shifted, mask_index_shifted, x_shifted, block_end_shifted, num_transfer_tokens[:, i],
                temperature, top_p, top_k, remasking, minimal_topk, threshold, opt_softmax
                )
                i+=1

                x[:, 1:][transfer_index_shifted] = x0_shifted[transfer_index_shifted]

                # early stop: with eos_early_stop tag & + eos token appeared + this block has been unmasked
                if eos_early_stop and (x[:, 1:][transfer_index_shifted] == eos_id).any():
                    eos_pos = (x[0, :] == eos_id).nonzero(as_tuple=True)[0][0]
                    if not (x[:, current_block_start:eos_pos]==mask_id).any():
                        # print("return 0")
                        # print(model.tokenizer.decode(x[0]))
                        return x
                    else:
                        x[:, eos_pos+1:] = eos_id
        
        # print("return 1")
        # print(model.tokenizer.decode(x[0]))
        return x

    @staticmethod
    def _top_k_logits(logits, k):
        if k is None or k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(
                logits < min_values, torch.full_like(logits, float("-inf")), logits
            )

    @staticmethod
    def _top_p_logits(logits, p):
        if p is None or p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        
        # Greedy mode: temperature = 0, no top-k/p
        if temperature == 0.0 and (top_k in (None, 0)) and (top_p is None or top_p >= 1.0):
            probs = F.softmax(logits, dim=-1)
            token = logits.argmax(dim=-1, keepdim=True)
            token_prob = probs.gather(-1, token)
            return token.view(*orig_shape), token_prob.view(*orig_shape)
        
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.tensor([], dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens


    @torch.no_grad()
    def generate_bd(
        self,
        data: Optional[torch.Tensor] = None,
        temperature: int = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = True,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
    ):
        r"""
        Generates tokens using a block-wise, iterative refinement strategy.
        This method operates differently from standard autoregressive generation. It first creates a template of the
        full desired length, filled with a special `mask_id`. It then processes this template in segments (`blocks`)
        and iteratively "denoises" or "refines" the `mask_id` tokens into actual tokens over a series of `steps` for
        each block. A custom block-diagonal causal attention mask ensures that generation within a block can attend to
        all previous blocks but not future ones.
        <Tip warning={true}>
        This is a specialized generation method. The quality and speed of the output are highly dependent on the interplay
        between `block_length`, `steps`, and `threshold`. It aims to achieve faster generation through parallel
        decoding within blocks, which is a departure from the token-by-token generation of standard `.generate()` methods.
        </Tip>
        Parameters:
            inputs (`torch.Tensor`):
                The token sequence used as a prompt for the generation.
            temperature (`float`, *optional*, defaults to 0.0):
                The value used to module the next token probabilities. A value of 0.0 corresponds to greedy decoding.
            block_length (`int`, *optional*, defaults to 32):
                The size of each generation block. The model generates text in parallel within these blocks. This is a
                key parameter for controlling the granularity of the generation process.
            steps (`int`, *optional*, defaults to 32):
                The number of iterative refinement (or "denoising") steps to perform for each block. Within each block,
                the model will try to replace `mask_id` tokens with real tokens for this many iterations.
            gen_length (`int`, *optional*, defaults to 2048):
                The maximum number of tokens to generate, excluding the prompt.
            top_p (`float`, *optional*):
                If set to a float value between 0 and 1, only the most probable tokens with probabilities that add up to
                `top_p` or higher are kept for generation (nucleus sampling).
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            eos_early_stop (`bool`, *optional*, defaults to `False`):
                If `True`, generation will stop as soon as a valid End-Of-Sequence token is generated and confirmed,
                even if `gen_length` has not been reached.
            minimal_topk (`int`, *optional*, defaults to 1):
                A parameter used to dynamically adjust the number of refinement `steps`. The effective number of steps
                is capped at `gen_length // minimal_topk`.
            threshold (`float`, *optional*, defaults to 0.95):
                The confidence probability threshold for accepting a sampled token. During each refinement step, a
                sampled token is only kept if its probability is above this threshold. If not enough tokens meet the
                threshold, the ones with the highest confidence are chosen.
            eos_id (`int`, *optional*, defaults to 156892):
                The token ID for the end-of-sequence token. Used for `eos_early_stop`.
            mask_id (`int`, *optional*, defaults to 156895):
                The token ID used as a placeholder for tokens that are yet to be generated. This is central to the
                iterative refinement algorithm.
        Return:
            `torch.Tensor`: A string containing the generated token IDs, starting
            after the prompt and stopping at the first `eos_id` or `gen_length`.
        """
        steps = min(steps, gen_length // minimal_topk)
        # input_ids = inputs.to(self.device)
        input_ids = data['input_ids']

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))

        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).bool()

        # log().to(torch.bfloat16)  .bool()

        # block_diffusion_attention_mask = torch.where(
        #     block_diffusion_attention_mask, 0.0, float("-inf")
        # ).to(torch.bfloat16)

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True

        prefill_blocks = prompt_length // block_length

        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, denoising_steps_per_block
        )
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            # cur_attn_mask = block_diffusion_attention_mask[
            #     :, 0, current_window_end-1, :current_window_end
            # ]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            # cur_attn_mask = block_diffusion_attention_mask[
            #     :, :current_window_end
            # ]
            cur_position_ids = position_ids[:, :current_window_end]

            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break
                data["attention_mask"] = cur_attn_mask
                data["position_ids"] = cur_position_ids
                data['input_ids'] = cur_x
                # logits = self.forward(
                #     cur_x,
                #     attention_mask=cur_attn_mask,
                #     position_ids=cur_position_ids,
                # ).logits
                logits = self(**data, z_loss=False).logits

                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(
                        confidence[0],
                        k=min(num_to_transfer, active_block_mask.sum().item()),
                    )
                    transfer_index[0, idx] = True

                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, : eos_pos + 1]
                            return final_x

            x[:, :current_window_end] = cur_x
            if (
                eos_id is not None
                and (x[0, prompt_length:current_window_end] == eos_id).any()
            ):
                break

        generated_answer = x[:, : prompt_length + gen_length]

        mask_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, : input_ids.shape[1] + first_mask_position + 1]

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_llada2_moe(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    使用高效内核 liger_rotary_pos_emb 实现的llada2_moe版 apply_rotary_pos_emb。
    """
    # 步骤 1: 扩展 cos 和 sin 的维度以进行广播，这部分逻辑不变
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 步骤 2: 确定旋转维度，并将 q 和 k 分割为旋转部分和非旋转部分，这部分逻辑不变
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # 步骤 3: 【核心替换点】
    # 将原来的手动计算替换为对旋转部分（q_rot, k_rot）调用高效内核
    q_embed_rot, k_embed_rot = liger_rotary_pos_emb(q_rot, k_rot, cos, sin)

    # 步骤 4: 将旋转后的结果与之前未动过的部分拼接回来，这部分逻辑不变
    q_embed = torch.cat([q_embed_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_embed_rot, k_pass], dim=-1)
    
    return q_embed, k_embed


# if is_liger_kernel_available():
#     apply_rotary_pos_emb = apply_rotary_pos_emb_llada2_moe
#     LLaDA2MoeRMSNorm = LigerRMSNorm
#     logger.info_rank0("Apply liger kernel to LLaDA2Moe")


__all__ = ["LLaDA2MoE_VLForConditionalGeneration", "LLaDA2MoE_VLModel", "LLaDA2MoePreTrainedModel", "LLaDA2MoeModel"]
# ==================== inline: data/llada2_vl_sft_datasets_bd.py::preprocess_qwen_2_visual ====================
import copy
from typing import Dict
IGNORE_INDEX = -100

def preprocess_qwen_2_visual(
    data,
    tokenizer,
    grid_thw_image=[],
    grid_thw_video=[],
    max_length=8192,
) -> Dict:
    system_message = "You are a helpful assistant."

    # ⚠️ 必须与训练 (llada2_vl_sft_bd_datasets.py) 完全一致地覆盖 chat_template，
    # 否则 ckpt 自带模板会插入 'detailed thinking off<|role_end|>' 等训练未见过的 token，
    # 导致 prompt 序列与训练分布不一致、生成乱码。
    chat_template = "{%- if messages and messages[0].role == 'system' %}{{- '<role>SYSTEM</role>' + messages[0].content + '\\n' }}{%- endif %}{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{%- else %}{%- set content = '' %}{%- endif %}{%- if message.role == 'user' %}{{- '<role>HUMAN</role>' + content + '<|role_end|>' }}{%- elif message.role == 'assistant' %}{{- '<role>ASSISTANT</role>' + content + '<|role_end|>' }}{%- endif %}{%- endfor %}"
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    input_ids += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])

    targets += [IGNORE_INDEX] * len(input_ids)

    user_contents = data['user_prompt']
    assistant_contents = data['answer_text']
    for i in range(len(user_contents)):
        user_content = user_contents[i]
        assistant_content = assistant_contents[i]

        if "<image>" in user_content:
            parts = user_content.split("<image>")
            new_parts = []
            for j in range(len(parts) - 1):
                new_parts.append(parts[j])
                replacement = (
                    "<|vision_start|>"
                    + f"<|image_pad|>"
                    * grid_thw_image[visual_replicate_index_image]
                    + "<|vision_end|>"
                )
                new_parts.append(replacement)
                visual_replicate_index_image += 1
            new_parts.append(parts[-1])
            user_content = "".join(new_parts)

        elif i == 0 and visual_replicate_index_image == 0 and grid_thw_image is not None and len(grid_thw_image) > 0:
            user_content = "<image>" * len(grid_thw_image) + "\n" + user_content

            parts = user_content.split("<image>")
            new_parts = []
            for j in range(len(parts) - 1):
                new_parts.append(parts[i])
                replacement = (
                    "<|vision_start|>"
                    + f"<|image_pad|>"
                    * grid_thw_image[visual_replicate_index_image]
                    + "<|vision_end|>"
                )
                new_parts.append(replacement)
                visual_replicate_index_image += 1
            new_parts.append(parts[-1])
            user_content = "".join(new_parts)

        user_conv = [{"role": "user", "content": user_content}]
        encode_id = tokenizer.apply_chat_template(user_conv)
        input_ids += encode_id
        targets += [IGNORE_INDEX] * len(encode_id)


        assistant_conv = [{"role": "assistant", "content": assistant_content}]
        encode_id = tokenizer.apply_chat_template(assistant_conv)
        input_ids += encode_id
        target_mask = encode_id.copy()
        # target_mask[:3] = [IGNORE_INDEX] * 3
        target_mask[:5] = [IGNORE_INDEX] * 5
        targets += target_mask


    assert len(input_ids) == len(targets), f"{len(input_ids)} != {len(targets)}"

    input_ids = input_ids + [tokenizer.eos_token_id]
    targets = targets + [tokenizer.eos_token_id]

    input_ids = input_ids[:max_length]
    targets = targets[:max_length]


    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# ============================ 推理 + 可验证 demo ============================
import argparse
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

# 与评测口径完全一致：point 模板 / block-diffusion 特殊 token / 坐标解析
POINT_TPL = ("Output the center point of the position corresponding to the following instruction:\n"
             "{instr}\n\n"
             "The output should just be the coordinates of a point, in the format [x,y]. "
             "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
             "the output should be [-1,-1].")
MASK_ID, EOS_ID = 156895, 156892
COORD_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*"
    r"(?:,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*)?\]")


def parse_point(text):
    """模型输出 4 数 bbox 取中心 /1000 -> 0~1；退而 2 数 point；[-1,-1] 判 wrong。与 eval_ssv2 同。"""
    if not text:
        return None
    m = list(COORD_RE.finditer(text))
    if not m:
        return None
    g = m[-1].groups()
    nums = [float(x) for x in g if x is not None]
    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        if x1 == -1 and y1 == -1:
            return [-1, -1]
        return [(x1 + x2) / 2 / 1000.0, (y1 + y2) / 2 / 1000.0]
    if len(nums) >= 2:
        x, y = nums[:2]
        if x == -1 and y == -1:
            return [-1, -1]
        return [x / 1000.0, y / 1000.0]
    return None


def hit(point, bbox, img_wh):
    """point(0~1) 落 GT bbox(像素)内 = correct。"""
    if point is None or point == [-1, -1]:
        return False
    w, h = img_wh
    bx1, by1, bx2, by2 = bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h
    return (bx1 <= point[0] <= bx2) and (by1 <= point[1] <= by2)


def build_batch(img_path, instruction, image_processor, tokenizer, max_length=8192):
    image = Image.open(img_path).convert("RGB")
    vp = image_processor.preprocess(image, return_tensors="pt")
    pixel_values = vp["pixel_values"]
    grid_thw = vp["image_grid_thw"][0]
    grid_thw_merged = [int(grid_thw.prod() // image_processor.merge_size ** 2)]
    # B 格式：image 占位放 prompt 文字末尾（训练同序，比 image 在前高 ~17 点）
    prompt = POINT_TPL.format(instr=instruction)
    data = {"user_image": [img_path], "user_prompt": [f"{prompt}\n<image>"], "answer_text": [""]}
    dd = preprocess_qwen_2_visual(data, tokenizer, grid_thw_image=grid_thw_merged,
                                 grid_thw_video=[], max_length=max_length)
    dd["position_ids"] = None
    dd["pixel_values"] = pixel_values
    dd["image_grid_thw"] = grid_thw.unsqueeze(0)
    dd["attention_mask"] = None
    mb = {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in dd.items()}
    qnum = int((mb["labels"] == -100).sum())
    mb["input_ids"] = mb["input_ids"][:, :qnum]
    return mb, qnum, grid_thw


def load(ckpt):
    max_pixels = int(os.environ.get("IMAGE_MAX_PIXELS", 1280 * 28 * 28))
    print(f"[env] torch={torch.__version__} cuda_ok={torch.cuda.is_available()} IMAGE_MAX_PIXELS={max_pixels}")
    # image_processor 直接从 ckpt 取（unpacked ckpt 自带 preprocessor_config.json），无需外部 qwen 目录
    image_processor = AutoImageProcessor.from_pretrained(ckpt)
    image_processor.max_pixels = max_pixels
    image_processor.min_pixels = 20 * 28 * 28
    image_processor.size["longest_edge"] = 1024
    image_processor.size["shortest_edge"] = 28
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    print(f"[load] model(bf16) <- {ckpt}")
    model = LLaDA2MoE_VLForConditionalGeneration.from_pretrained(ckpt, device_map="cuda").to(torch.bfloat16).eval()
    print("[load] OK")
    return image_processor, tokenizer, model


def infer_one(model, tokenizer, image_processor, img_path, instruction,
              max_length=8192, gen_length=32, steps=32, block_length=32):
    mb, qnum, grid_thw = build_batch(img_path, instruction, image_processor, tokenizer, max_length)
    n_img_tokens = int(grid_thw.prod() // (image_processor.merge_size ** 2))
    with torch.no_grad():
        out = model.generate_bd(data=mb, block_length=block_length, steps=steps,
                                gen_length=gen_length, temperature=1.0, threshold=0.99)
    raw = out[0][qnum:]
    text = tokenizer.decode([t for t in raw.tolist() if t not in (MASK_ID, EOS_ID)],
                            skip_special_tokens=True).strip()
    return text, n_img_tokens


def main():
    ap = argparse.ArgumentParser(description="LLaDA2-VL HF grounding 推理 / SSv2 demo（自包含）")
    ap.add_argument("--ckpt", required=True, help="unpacked MoE HF ckpt 目录")
    ap.add_argument("--image", default="", help="单图路径；留空走 --verify-ssv2")
    ap.add_argument("--prompt", default="click the search button", help="自然语言指令")
    ap.add_argument("--max-length", type=int, default=8192, help="输入截断；4K 大图空输出时调到 20000+")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--verify-ssv2", action="store_true", help="跑真实 SSv2 样本，解析坐标对照 GT")
    ap.add_argument("--ssv2-imgs", default="", help="ScreenSpot-V2 image directory")
    ap.add_argument("--ssv2-json", default="", help="ScreenSpot-V2 annotation directory")
    ap.add_argument("--platform", default="desktop", choices=["desktop", "mobile", "web"])
    ap.add_argument("--n", type=int, default=3, help="verify 模式取前 N 条")
    args = ap.parse_args()

    image_processor, tokenizer, model = load(args.ckpt)

    # ---- 模式 A：单图推理 ----
    if not args.verify_ssv2:
        if not args.image:
            ap.error("--image 必填（或加 --verify-ssv2 跑 SSv2 样本）")
        text, n_tok = infer_one(model, tokenizer, image_processor, args.image, args.prompt,
                                max_length=args.max_length, gen_length=args.gen_length,
                                steps=args.steps, block_length=args.block_length)
        point = parse_point(text)
        print("=" * 60)
        print("IMAGE :", args.image)
        print("PROMPT:", args.prompt)
        print("img_tokens:", n_tok)
        print("GENERATION:", text)
        print("POINT(0~1):", point)
        print("=" * 60)
        raise SystemExit(0 if text.strip() else 1)

    # ---- 模式 B：SSv2 可验证 demo（真实样本，解析+对照 GT）----
    if not args.ssv2_imgs or not args.ssv2_json:
        ap.error("--verify-ssv2 requires both --ssv2-imgs and --ssv2-json")
    jf = os.path.join(args.ssv2_json, f"{args.platform}.json")
    if not os.path.exists(jf):
        ap.error(f"SSv2 json 不存在: {jf}（verify 模式需要 SSv2 数据集）")
    samples = json.load(open(jf))[: args.n]
    print(f"[verify] {args.platform}.json -> {len(samples)} samples")
    correct, parse_ok = 0, 0
    for i, s in enumerate(samples):
        fname = s.get("img_filename") or s.get("image_path")
        img_path = os.path.join(args.ssv2_imgs, fname)
        instruction = s.get("instruction")
        bbox = s.get("bbox", [-1, -1, -1, -1])
        with Image.open(img_path) as im:
            img_wh = im.size
        t0 = time.time()
        text, n_tok = infer_one(model, tokenizer, image_processor, img_path, instruction,
                                max_length=args.max_length, gen_length=args.gen_length,
                                steps=args.steps, block_length=args.block_length)
        point = parse_point(text)
        ok = hit(point, bbox, img_wh)
        parse_ok += int(point is not None)
        correct += int(ok)
        print(f"\n[{i+1}/{len(samples)}] img={fname} img_tokens={n_tok} ({time.time()-t0:.1f}s)")
        print(f"    instruction : {instruction}")
        print(f"    raw_gen     : {text!r}")
        print(f"    pred_point  : {point}   gt_bbox={bbox}   img_wh={img_wh}")
        print(f"    -> {'CORRECT' if ok else 'WRONG'}")
    print("=" * 60)
    print(f"PIPELINE : {'OK' if parse_ok >= 1 else 'FAIL(无可解析坐标)'}  (parseable {parse_ok}/{len(samples)})")
    print(f"ACCURACY : {correct/len(samples):.3f}  ({correct}/{len(samples)})  <- 小样本趋势, 全量请自行循环")
    print("=" * 60)
    raise SystemExit(0 if parse_ok >= 1 else 1)


if __name__ == "__main__":
    main()

# ============================ 环境要求（pip 依赖）============================
#   torch==2.5.1+cu124   transformers==4.49.0 或 4.51.0   flash-attn==2.7.4.post1
#   Pillow   numpy   einops   accelerate   sentencepiece   protobuf   safetensors
# 安装：
#   pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
#   pip install transformers==4.51.0 Pillow numpy einops accelerate sentencepiece protobuf safetensors
#   pip install ninja && pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
#
# --- ckpt 要求 ---
#   必须是 *unpacked MoE* 的 HF 目录（hf_ckpt_unpacked），packed 直接加载会 256 专家丢弃 -> 乱码。
#   该目录需自带 preprocessor_config.json / tokenizer* / config.json / *.safetensors。
#   分辨率档 IMAGE_MAX_PIXELS 必须匹配训练档：highres=12845056 / default=1003520。
#   flash_attn 是模型代码硬 import，无法省略，且需与 torch 版本对齐后源码编译。
