import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
from torch.utils.data import DataLoader

from jaxtyping import Float, Int, Bool
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# import pysvelte

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)


def patch_masked_residue(
    target_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    mask: Bool[torch.Tensor, "batch pos"],
    source_cache,
):
    device = target_residual_component.device
    target_residual_component[mask.to(device), :] = (
        source_cache[hook.name].to(device)[mask.to(device), :].to(device)
    )
    return target_residual_component


def intervene_stream_masked(
    model,
    source_tokens,
    target_tokens,
    source_mask: Bool[torch.Tensor, "batch layer pos"],
    target_mask: Bool[torch.Tensor, "batch layer pos"],
):
    names_filter = lambda x: x in ["resid_pre", "resid_post", "rot_k"]
    device = model.cfg.device
    source_logits, source_cache = model.run_with_cache(
        source_tokens, names_filter=names_filter, device=device
    )
    target_logits, target_cache = model.run_with_cache(
        target_tokens, names_filter=names_filter, device=device
    )
    hooks = []
    inv_target_mask = torch.logical_not(target_mask)
    for layer in range(model.cfg.n_layers):
        hook_fn = partial(
            patch_masked_residue,
            mask=source_mask[:, layer, :],
            source_cache=source_cache,
        )
        hooks.append((utils.get_act_name("resid_pre", layer), hook_fn))
        post_hook = partial(
            patch_masked_residue,
            mask=inv_target_mask[:, layer, :],
            source_cache=target_cache,
        )
        hooks.append((utils.get_act_name("resid_post", layer), post_hook))
    intervened_logits = model.run_with_hooks(
        target_tokens, fwd_hooks=hooks, return_type="logits"
    )
    return intervened_logits


def patch_residue(
    target_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    target_source: Int[torch.Tensor, "batch pos"],
    mask: Bool[torch.Tensor, "batch pos"],
    source_cache,
):
    device = target_residual_component.device
    target_residual_component[mask.to(device), :] = torch.gather(
        source_cache[hook.name].to(device),  # batch source_pos d_model
        1,  # source_pos
        einops.repeat(
            target_source.to(device),
            "b p -> b p d",
            d=target_residual_component.shape[2],
        ),
    )[mask.to(device), :]
    return target_residual_component


def intervene_stream(
    model,
    source_tokens,
    target_tokens: Float[torch.Tensor, "batch pos"],
    target_independent_source: Int[torch.Tensor, "batch layer pos"],
    target_dependent_mask: Bool[torch.Tensor, "batch layer pos"],
    source_cache=None,
):
    """
    This assumes that the target_independent_mask satisfies:
        - for any layer l
        - for any two batches b1 and b2
        - sum(target_independent_mask[b1, l, :]) = sum(target_independent_mask[b2, l, :])
    """
    target_independent_source = torch.clone(target_independent_source)
    device = model.cfg.device
    if source_cache is None:
        source_logits, source_cache = model.run_with_cache(source_tokens, device=device)
    target_logits, target_cache = model.run_with_cache(target_tokens, device=device)
    hooks = []
    inv_target_dependent_mask = torch.logical_not(target_dependent_mask)
    target_independent_mask = target_independent_source != -1
    target_independent_source[torch.logical_not(target_independent_mask)] = 0
    for layer in range(model.cfg.n_layers):
        hook_fn = partial(
            patch_residue,
            target_source=target_independent_source[:, layer, :],
            mask=target_independent_mask[:, layer, :],
            source_cache=source_cache,
        )
        hooks.append((utils.get_act_name("resid_pre", layer), hook_fn))
        post_hook = partial(
            patch_masked_residue,
            mask=inv_target_dependent_mask[:, layer, :],
            source_cache=target_cache,
        )
        hooks.append((utils.get_act_name("resid_post", layer), post_hook))
    intervened_logits = model.run_with_hooks(
        target_tokens, fwd_hooks=hooks, return_type="logits"
    )
    return intervened_logits


def intervene_stream_masked_refactored(
    model,
    source_tokens,
    target_tokens,
    source_mask: Bool[torch.Tensor, "batch layer pos"],
    target_mask: Bool[torch.Tensor, "batch layer pos"],
):
    target_independent_source = torch.zeros_like(
        target_mask, dtype=torch.int64, device=model.cfg.device
    )
    target_independent_source[:, :, :] = torch.arange(
        0, target_mask.shape[2], dtype=torch.int64
    )
    target_independent_source[torch.logical_not(source_mask)] = -1
    return intervene_stream(
        model=model,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        target_independent_source=target_independent_source,
        target_dependent_mask=target_mask,
    )


def rotary_deltas(
    x: Float[torch.Tensor, "batch pos head_index d_head"],
    pos_deltas: Int[torch.Tensor, "batch pos"],
    attn,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # adapted from components.py -> Attention -> apply_rotary
    x_pos = x.size(1)
    x_rot = x[..., : attn.cfg.rotary_dim]  # batch pos head_index d_head
    x_pass = x[..., attn.cfg.rotary_dim :]
    x_flip = attn.rotate_every_two(x_rot)
    abs_pos = torch.abs(pos_deltas)
    coses = attn.rotary_cos[abs_pos]  # batch pos d_head
    sines = attn.rotary_sin[abs_pos] * torch.sign(pos_deltas)[..., None]
    x_rotated = x_rot * einops.rearrange(
        coses, "b p d -> b p 1 d"
    ) + x_flip * einops.rearrange(sines, "b p d -> b p 1 d")
    return torch.cat([x_rotated, x_pass], dim=-1)


def patch_rotary_k(
    target_k: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    pos_deltas: Int[torch.Tensor, "batch pos"],
    rotate_function,
):
    # consistency tests:
    # y = rotate_function(target_k, pos_deltas)
    # x = rotate_function(y, -pos_deltas)
    # assert torch.allclose(target_k, x, rtol=1e-3, atol=1e-4)
    return rotate_function(target_k, pos_deltas.to(target_k.device))


def binary_swap(tensor1, tensor2):
    tensor1 ^= tensor2
    tensor2 ^= tensor1
    tensor1 ^= tensor2


def swap_causal_mask(model, causal_mask: Bool[torch.Tensor, "layer pos pos"]):
    for layer in range(model.cfg.n_layers):
        binary_swap(
            model.blocks[layer].attn.mask[
                : causal_mask.shape[1], : causal_mask.shape[2]
            ],
            causal_mask[layer],
        )


from contextlib import contextmanager


@contextmanager
def impose_causal_mask(model, causal_mask: Bool[torch.Tensor, "layer pos pos"]):
    swap_causal_mask(model, causal_mask)
    yield None
    swap_causal_mask(model, causal_mask)


def NAMES_FILTER(no_rotate):
    if no_rotate:
        return lambda x: any(y in x for y in ["resid_pre", "resid_post"])
    else:
        return lambda x: any(y in x for y in ["resid_pre", "resid_post", "rot_k"])


def intervene_stream_rotary(
    model,
    source_tokens,
    target_tokens: Float[torch.Tensor, "batch pos"],
    target_independent_source: Int[torch.Tensor, "batch layer pos"],
    target_dependent_mask: Bool[torch.Tensor, "batch layer pos"],
    source_cache=None,
    no_rotate=False,
    pos_deltas=None,
):
    """
    This assumes that the target_independent_mask satisfies:
        - for any layer l
        - for any two batches b1 and b2
        - sum(target_independent_mask[b1, l, :]) = sum(target_independent_mask[b2, l, :])

    Rotation conventions:
        - we're assuming that there exists some kind of "position invariant truth"
        that the network tries to learn
            - e.g. in the construction of the keys, you take the original keys
            and rotate by exp(i * source_pos * omega)
        - to recover the "position invariant truth", when you transplant the original keys
        to the target, you need to rotate keys by exp(i * (source_pos - target_pos) * omega)

    pos_deltas is applied regardless of masks!! Explicitly set to zero if you want to not apply it

    Args:
        pos_deltas: Int[torch.Tensor, "batch layer pos"] - source position - target position
    """
    target_independent_source = torch.clone(target_independent_source)
    device = model.cfg.device
    if source_cache is None:
        source_logits, source_cache = model.run_with_cache(
            source_tokens, names_filter=NAMES_FILTER(no_rotate), device=device
        )
    target_logits, target_cache = model.run_with_cache(
        target_tokens, names_filter=NAMES_FILTER(no_rotate), device=device
    )
    hooks = []
    inv_target_dependent_mask = torch.logical_not(target_dependent_mask)
    target_independent_mask = target_independent_source != -1
    target_independent_source[torch.logical_not(target_independent_mask)] = 0
    if pos_deltas is None:
        default_positions = einops.repeat(
            torch.arange(target_independent_source.shape[2], device=device),
            "p -> b l p",
            b=target_independent_source.shape[0],
            l=target_independent_source.shape[1],
        )
        pos_deltas = target_independent_source - (
            target_independent_mask * default_positions
        )

    if no_rotate:
        pos_deltas *= 0
    for layer in range(model.cfg.n_layers):
        hook_fn = partial(
            patch_residue,
            target_source=target_independent_source[:, layer, :],
            mask=target_independent_mask[:, layer, :],
            source_cache=source_cache,
        )
        hooks.append((utils.get_act_name("resid_pre", layer), hook_fn))
        if not no_rotate:
            hook_k = partial(
                patch_rotary_k,
                pos_deltas=pos_deltas[:, layer, :],
                rotate_function=partial(rotary_deltas, attn=model.blocks[layer].attn),
            )
            hooks.append((utils.get_act_name("rot_k", layer), hook_k))

        post_hook = partial(
            patch_masked_residue,
            mask=inv_target_dependent_mask[:, layer, :],
            source_cache=target_cache,
        )
        hooks.append((utils.get_act_name("resid_post", layer), post_hook))
    intervened_logits = model.run_with_hooks(
        target_tokens, fwd_hooks=hooks, return_type="logits"
    )
    return intervened_logits
