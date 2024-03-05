import os
import copy
import warnings
from coref import COREF_ROOT
from functools import partial

import numpy as np
import torch
from einops import rearrange, einsum, repeat
from transformer_lens import HookedTransformer
from coref.experiments.common import pretty_print_logits, lookup_answer_logits

from transformer_lens.utilities.devices import get_device_for_block_index
import transformer_lens.utils as utils

import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from coref.interventions import patch_masked_residue, patch_rotary_k, rotary_deltas


from coref.datasets.common import Name, Attr, rotate
import coref.datasets.capitals as capitals
from coref.datasets.common import (
    Statement,
    stack_tokens,
    default_mcq_labels,
    generate_shuffler,
)
from coref.datasets.api import generate_tokenized_prompts, generate_options


def compose_hook_fns(hook_fns):
    def wrapper(tensor, hook, **kwargs):
        for hook_fn in hook_fns:
            tensor = hook_fn(tensor, hook, **kwargs)
        return tensor

    return wrapper


def inject_countries_hook_fn(act, hook, *, param, selector, side=None):
    param = param.to(act.device)
    if side is None:
        target_1 = selector["countries"][0]
        target_2 = selector["countries"][1]
        act[:, target_1, :] += param
        act[:, target_2, :] -= param
    else:
        target = selector["countries"][side]
        act[:, target, :] += param
    return act


def inject_names_hook_fn(act, hook, *, param, width=2, selector, side=None):
    param = param.to(act.device)
    if side is None:
        target_1 = selector["names"][0]
        target_2 = selector["names"][1]
        act[:, target_1 : target_1 + width, :] += param
        act[:, target_2 : target_2 + width, :] -= param
    else:
        target = selector["names"][side]
        act[:, target : target + width, :] += param

    return act


def inject_hook_fn(act, hook, *, param, category, width, selector, side=None):
    param = param.to(act.device)
    if side is None:
        target_1 = selector[category][0]
        target_2 = selector[category][1]
        act[:, target_1 : target_1 + width, :] += param
        act[:, target_2 : target_2 + width, :] -= param
    else:
        target = selector[category][side]
        act[:, target : target + width, :] += param
    return act


def general_inject_hook_fn(act, hook, *, param, width, token_maps, extractor):
    param = param.to(act.device)
    if len(param.shape) == 2:
        param = repeat(param, "width dim -> batch width dim", batch=act.shape[0])
    else:
        assert param.shape[0] == act.shape[0]
        assert len(param.shape) == 3
    targets = extractor(token_maps)
    for batch in range(act.shape[0]):
        act[batch, targets[batch] : targets[batch] + width, :] += param[batch]
    return act


def project(act, param):
    """
    act: (batch, length, dim)
    param: (subspace, length, dim)

    Returns: (batch, length, dim)
    """
    return einsum(
        (
            einsum(
                act,
                param,
                "batch length dim, subspace length dim -> batch subspace length",
            )
            / (
                einsum(
                    param,
                    param,
                    "subspace length dim, subspace length dim -> subspace length",
                )
                + 1e-8
            )
        ),
        param,
        "batch subspace length, subspace length dim -> batch length dim",
    )


def orthogonalize_param(param):
    """
    param: (subspace, length, dim)
    """
    ret_param = param.clone()
    for i in range(1, ret_param.shape[0]):
        ret_param[[i], :, :] -= project(param[[i], :, :], param[:i, :, :])
    return ret_param


def project_hook_fn(
    act, hook, *, param, selector, coefficients, width, category, orthogonalize=False
):
    param = param.to(act.device)
    if orthogonalize:
        param = orthogonalize_param(param)
    targets = selector[category]  # num_entities implicit from len(targets)
    acts = [
        project(act[:, target : target + width, :], param) for target in targets
    ]  # (num_entities, batch, width, dim)
    # assert len(targets) == 3 and len(coefficients) == 3
    act = act.clone()
    # matrix multiplication by hand
    for target, coeffs in zip(targets, coefficients):
        # assert len(coeffs) == 3
        act[:, target : target + width, :] += sum(
            coeff * act for coeff, act in zip(coeffs, acts)
        )
    return act


project_countries_hook_fn = partial(project_hook_fn, category="countries", width=1)
project_names_hook_fn = partial(project_hook_fn, category="names", width=2)


def swap_hook_fn(act, hook, *, param, selector, width, category):
    act_out = act.clone()
    act_out[:, selector[category][0] : selector[category][0] + width, :] = act[
        :, selector[category][1] : selector[category][1] + width, :
    ]
    act_out[:, selector[category][1] : selector[category][1] + width, :] = act[
        :, selector[category][0] : selector[category][0] + width, :
    ]
    return act_out


def overwrite_hook_fn(act, hook, param, selector, category, width, side):
    param = param.to(act.device)
    target = selector[category][side]
    act[:, target : target + width, :] = param
    return act


def general_overwrite_hook_fn(act, hook, param, width, token_maps, extractor):
    """
    param: [batch, width, dim]
    """
    param = param.to(act.device)
    if len(param.shape) == 2:
        param = repeat(param, "width dim -> batch width dim", batch=act.shape[0])
    else:
        assert param.shape[0] == act.shape[0]
        assert len(param.shape) == 3
    targets = extractor(token_maps)
    for batch in range(act.shape[0]):
        act[batch, targets[batch] : targets[batch] + width, :] = param[batch]
    # print('overwriting', targets)
    return act


def id_hook_fn(act, hook):
    return act


def build_selector(vocab, num_entities):
    prompt_id = 10
    tokens, token_map = generate_tokenized_prompts(
        vocab=vocab,
        context=[Statement(i, i, "normal") for i in range(num_entities)],
        query_name=Name.b,
        prompt_id=prompt_id,
    )
    return {
        "names": [statement["subject"].start for statement in token_map["context"]],
        "countries": [statement["country"].start for statement in token_map["context"]],
        "dependent_start": token_map["context_section"].end,
    }


global_prompt_id_counter = 0


def reorder_answers(all_answers, all_query_names):
    """
    all_answers : (batch_size, num_entities)
    all_query_names : (batch_size,)

    Modifies all_answers in place
    """
    for r in range(all_answers.shape[0]):
        t = all_answers[r, 0]
        all_answers[r, 0] = all_answers[r, all_query_names[r]]
        all_answers[r, all_query_names[r]] = t


def build_example(*, batch_size, vocab, num_entities, fixed_query_name=None):
    """
    Warning: This doesn't work for num_entities != 2
    """
    all_tokens, all_answers, all_query_names, token_maps = build_example_refactored(
        batch_size=batch_size,
        vocab=vocab,
        num_entities=num_entities,
        fixed_query_name=fixed_query_name,
    )
    with torch.no_grad():
        for r in range(batch_size):
            t = all_answers[r, 0].item()
            all_answers[r, 0] = all_answers[r, all_query_names[r]].item()
            all_answers[r, all_query_names[r]] = t
    return all_tokens, all_answers


def build_example_refactored(
    *,
    batch_size,
    vocab,
    num_entities,
    fixed_query_name=None,
    raw_query_name=None,
    context=None,
    prompt_id_start=None,
    template_context=None,
    template_type="normal",
    num_answers=None,
):
    """
    Warning: This is a breaking change.
    all_answers no longer gives the correct answer in the first column
    Args:
        fixed_query_name : None | Name
        raw_query_name : None | str

    Returns:
        all_tokens: (batch_size, length)
        all_answers: (batch_size, num_entities)
        all_query_names: (batch_size) - this is in terms of pseudo-name
        all_query_names: (batch_size)
    """
    if context is None:
        context = vocab.default_context(num_entities)
    if num_answers is None:
        num_answers = num_entities
    assert raw_query_name is None or fixed_query_name is None, "Only one can be set"
    global global_prompt_id_counter
    all_answers = []
    all_tokens = []
    all_query_names = []
    all_token_maps = []
    for i in range(batch_size):
        if prompt_id_start is None:
            global_prompt_id_counter += 1
            prompt_id = global_prompt_id_counter
        else:
            prompt_id = i + prompt_id_start
        if template_context is None and fixed_query_name is None:
            query_name = vocab.global_rng.integers(0, num_entities)
        else:
            query_name = fixed_query_name
        tokens, token_map = generate_tokenized_prompts(
            vocab=vocab,
            context=context,
            prompt_id=prompt_id,
            query_name=query_name,
            raw_query_name=raw_query_name,
            template_type=template_type,
            template_context=template_context,
        )
        all_token_maps.append(token_map)
        answers = [
            vocab.encode_single_word(
                vocab.fetch_shuffled_attr(answer_attr, prompt_id)[1]
            )
            for answer_attr in range(num_answers)
        ]
        all_answers.append(answers)
        all_tokens.append(tokens)
        all_query_names.append(
            template_context["query_name"]
            if template_context is not None
            else query_name
        )
    all_answers = torch.tensor(all_answers)
    all_tokens = stack_tokens(vocab, all_tokens)
    all_query_names = torch.tensor(all_query_names)
    all_token_maps = rotate(
        lambda substr_list: torch.tensor([[sub[0], sub[1]] for sub in substr_list])
    )(all_token_maps)
    return all_tokens, all_answers, all_query_names, all_token_maps


# def build_example_mcq(
#     *,
#     batch_size,
#     vocab,
#     num_entities,
#     fixed_query_name=None,
#     raw_query_name=None,
#     context=None,
#     prompt_id_start=None,
#     template_type='normal',
#     labels=default_mcq_labels,
# ):
#     """
#     Warning: This is a breaking change.
#     all_answers no longer gives the correct answer in the first column
#     Args:
#         fixed_query_name : None | Name
#         raw_query_name : None | str

#     Returns:
#         all_tokens: (batch_size, length)
#         all_answers: (batch_size, num_entities)
#         all_query_names: (batch_size)
#     """
#     if context is None:
#         context = vocab.default_context(num_entities)
#     assert raw_query_name is None or fixed_query_name is None
#     assert num_entities == len(context)
#     global global_prompt_id_counter
#     all_answers = []
#     all_tokens = []
#     all_query_names = []
#     all_token_maps = []
#     for i in range(batch_size):
#         if prompt_id_start is None:
#             global_prompt_id_counter += 1
#             prompt_id = global_prompt_id_counter
#         else:
#             prompt_id = i + prompt_id_start
#         if fixed_query_name is None:
#             query_name = vocab.global_rng.integers(0, num_entities)
#         else:
#             query_name = fixed_query_name
#         answer_strings = [
#             vocab.fetch_shuffled_attr(answer_attr, prompt_id)[1]
#             for answer_attr in range(num_entities)
#         ]
#         shuffler = generate_shuffler(prompt_id, num_entities)
#         option_string = generate_options(
#             labels=labels[:num_entities],
#             options=shuffler(answer_strings),
#         )
#         answer_tokens = [
#             vocab.encode_single_word(
#                 answer_label
#             )
#             for answer_label in shuffler(labels[:num_entities], invert=True)
#         ]

#         tokens, token_map = generate_tokenized_prompts(
#             vocab=vocab,
#             context=context,
#             prompt_id=prompt_id,
#             template_type=template_type,
#             template_context=dict(
#                 query_name=query_name,
#                 raw_query_name=raw_query_name,
#                 suffix=option_string
#             )
#         )
#         all_token_maps.append(token_map)
#         all_answers.append(answer_tokens)
#         all_tokens.append(tokens)
#         all_query_names.append(query_name)
#     all_answers = torch.tensor(all_answers)
#     all_tokens = stack_tokens(vocab, all_tokens)
#     all_query_names = torch.tensor(all_query_names)
#     all_token_maps = rotate(
#         lambda substr_list: torch.tensor([[sub[0], sub[1]] for sub in substr_list])
#     )(all_token_maps)
#     return all_tokens, all_answers, all_query_names, all_token_maps


class StaticInjectionParameters:
    def __init__(self, /, model, layer_start, layer_end):
        self.parameters = {
            "norms": {
                layer: torch.ones(
                    (),
                    device=get_device_for_block_index(layer, model.cfg),
                    requires_grad=True,
                )
                for layer in range(layer_start, layer_end)
            },
            "vector": torch.zeros(
                (model.cfg.d_model,),
                device=get_device_for_block_index(0, model.cfg),
                requires_grad=True,
            ),
        }

    def get_param_list(self):
        return list(self.parameters["norms"].values()) + [self.parameters["vector"]]

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.parameters["norms"]:
            return True, parameters.parameters["norms"][layer] * parameters.parameters[
                "vector"
            ].to(parameters.parameters["norms"][layer].device)
        else:
            return False, None

    @staticmethod
    def weight_loss(parameters, /, weight, norm_weight=0.1, dir_weight=0.01):
        norms = sum([v * v for v in parameters.parameters["norms"].values()]) / len(
            parameters.parameters["norms"]
        )
        dirs = torch.dot(
            parameters.parameters["vector"], parameters.parameters["vector"]
        )
        # return  norm_weight * norms + dir_weight * dirs
        return weight * norms * dirs


class UntiedInjectionParameters:
    def __init__(self, /, model, layer_start, layer_end, n_dim=1):
        self.parameters = {
            layer: torch.zeros(
                (n_dim, model.cfg.d_model),
                device=get_device_for_block_index(0, model.cfg),
                requires_grad=True,
            )
            for layer in range(layer_start, layer_end)
        }

    @classmethod
    def from_dict(cls, /, model, d):
        thing = cls(model=model, layer_start=0, layer_end=0)
        thing.parameters = d
        return thing

    def get_param_list(self):
        return list(self.parameters.values())

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.parameters:
            return True, parameters.parameters[layer]
        else:
            return False, None

    @staticmethod
    def batch_adapter(parameters, layer, prompt_ids, prompt_id_start):
        """
        prompt_ids are slices of absolute values
        """
        assert prompt_ids is not None
        return (
            True,
            parameters.parameters[layer][
                slice(
                    prompt_ids.start - prompt_id_start,
                    prompt_ids.stop - prompt_id_start,
                )
            ],
        )

    @staticmethod
    def duplicate_adapter(parameters, layer, tag, start, end):
        if start <= layer and layer < end:
            return True, parameters.parameters[tag]
        return False, None

    @staticmethod
    def tag_adapter(parameters, layer, tag):
        return True, parameters.parameters[tag]

    @staticmethod
    def weight_loss(parameters, /, weight):
        return (
            weight
            * sum([v.norm() ** 2 for v in parameters.parameters.values()])
            / len(parameters.parameters)
        )


class ScalingInjectionParameters:
    def __init__(self, /, model, d, n_dim=1, exp=True):
        """
        d: dict of layer -> (n_dim, model.cfg.d_model)
        """
        self.exp = exp
        self.d = d
        self.parameters = {
            layer: torch.zeros(
                (n_dim,),
                device=get_device_for_block_index(layer, model.cfg),
                requires_grad=True,
            )
            for layer in self.d.keys()
        }

    def get_param_list(self):
        return list(self.parameters.values())

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.parameters:
            return (
                True,
                (
                    torch.exp(parameters.parameters[layer][:, None])
                    if not hasattr(parameters, "exp") or parameters.exp
                    else parameters.parameters[layer][:, None]
                )
                * parameters.d[layer],
            )
        else:
            return False, None

    @staticmethod
    def weight_loss(parameters, /, weight):
        return (
            weight
            * sum([v.norm() ** 2 for v in parameters.parameters.values()])
            / len(parameters.parameters)
        )


class ResidualInjectionParameters:
    def __init__(self, /, model, layer_start, layer_end, n_dim=1):
        self.parameters = {
            layer: torch.zeros(
                (n_dim, model.cfg.d_model),
                device=get_device_for_block_index(0, model.cfg),
                requires_grad=True,
            )
            for layer in range(layer_start, layer_end)
        }
        self.layer_start = layer_start
        self.layer_end = layer_end

    def get_param_list(self):
        return list(self.parameters.values())

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.parameters:
            base = torch.zeros_like(parameters.parameters[parameters.layer_start])
            for i in range(parameters.layer_start, layer + 1):
                base += parameters.parameters[i]
            return True, base
        else:
            return False, None

    @staticmethod
    def duplicate_adapter(parameters, layer, tag, start, end):
        if start <= layer and layer < end:
            return True, parameters.parameters[tag]
        return False, None

    @staticmethod
    def tag_adapter(parameters, layer, tag):
        return True, parameters.parameters[tag]

    @staticmethod
    def weight_loss(parameters, /, weight):
        return (
            weight
            * sum([v.norm() ** 2 for v in parameters.parameters.values()])
            / len(parameters.parameters)
        )


class ResidualInjectionSubspace:
    def __init__(self, /, model, layer_start, layer_end, n_dim=1, n_subspace=1):
        self.parameters = {
            layer: torch.zeros(
                (n_subspace, n_dim, model.cfg.d_model),
                device=get_device_for_block_index(0, model.cfg),
                requires_grad=True,
            )
            for layer in range(layer_start, layer_end)
        }
        self.layer_start = layer_start
        self.layer_end = layer_end

    def get_param_list(self):
        return list(self.parameters.values())

    @classmethod
    def from_params(cls, /, model, param_list):
        self = cls(
            model=model,
            layer_start=param_list[0].layer_start,
            layer_end=param_list[0].layer_end,
        )
        self.parameters = {
            layer: torch.stack([p.parameters[layer] for p in param_list])
            for layer in range(self.layer_start, self.layer_end)
        }
        return self

    @staticmethod
    def default_adapter(parameters, layer):
        if layer in parameters.parameters:
            return True, sum(
                parameters.parameters[i]
                for i in range(parameters.layer_start, layer + 1)
            )
        else:
            return False, None

    @staticmethod
    def weight_loss(parameters, /, weight):
        # Sum of L2 norms across layers
        return (
            weight
            * sum([v.norm() for v in parameters.parameters.values()])
            / len(parameters.parameters)
        )


class CompoundParameters:
    def __init__(self, /, model, d):
        self.parameters = d

    def get_param_list(self):
        return list(k for i in self.parameters.values() for k in i.get_param_list())

    @staticmethod
    def default_adapter(parameters, layer, *, tag, child_adapter, prompt_ids=None):
        if tag in parameters.parameters:
            if "prompt_ids" in signature(child_adapter).parameters:
                return child_adapter(
                    parameters.parameters[tag], layer, prompt_ids=prompt_ids
                )
            else:
                return child_adapter(parameters.parameters[tag], layer)
        else:
            return False, None

    @staticmethod
    def weight_loss(parameters, **kwargs):
        weights = [v.weight_loss(v, **kwargs) for v in parameters.parameters.values()]
        device = weights[0].device
        return sum([v.to(device) for v in weights])


from inspect import signature


def apply_param_adapter(hook_fn, param_adapter):
    """
    hook_fn: (act, hook, param) -> act
    param_adapter: (parameters, layer) -> (valid, param)
    selector_constructor: (token_maps) -> selector: Dict[category: str -> tensor[batch]]
    Returns a hook_fn_generator :
            (parameters, layer, prompt_ids=None, token_maps=None)
                    -> hook_fn
    """

    def wrapper(parameters, layer, prompt_ids=None, token_maps=None):
        if "prompt_ids" in signature(param_adapter).parameters:
            valid, param = param_adapter(parameters, layer, prompt_ids=prompt_ids)
        else:
            valid, param = param_adapter(parameters, layer)
        if valid:
            if "token_maps" in signature(hook_fn).parameters:
                return partial(hook_fn, param=param, token_maps=token_maps)
            else:
                return partial(hook_fn, param=param)
        else:
            return id_hook_fn

    return wrapper


def compose_hook_fn_generators(hook_fn_generators):
    def wrapper(parameters, layer, prompt_ids=None, token_maps=None):
        return compose_hook_fns(
            [
                hook_fn_generator(
                    parameters, layer, prompt_ids=prompt_ids, token_maps=token_maps
                )
                for hook_fn_generator in hook_fn_generators
            ]
        )

    return wrapper


def stack_residual(residual_params):
    return torch.stack(
        [
            residual_params.default_adapter(residual_params, layer)[1]
            for layer in range(residual_params.layer_start, residual_params.layer_end)
        ]
    )


def run_cached_model(
    execution_fn,
    model,
    names_filter=None,
    device=None,
    remove_batch_dim=False,
    incl_bwd=False,
    reset_hooks_end=True,
    clear_contexts=False,
):
    """
    Modified from HookedRootModule.run_with_cache
    """
    cache_dict, fwd, bwd = model.get_caching_hooks(
        names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
    )

    with model.hooks(
        fwd_hooks=fwd,
        bwd_hooks=bwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        model_out = execution_fn()
        if incl_bwd:
            model_out.backward()

    return model_out, cache_dict


def run_intervened_model(
    model,
    parameters,
    source_tokens,
    hook_fn_generator,
    dependent_start,
    pos_deltas_fn=None,
):
    """
    Args:
        hook_fn_generator : (parameters, layer) -> hook_fn
        pos_deltas_fn: (pos_deltas) -> pos_deltas (Int[torch.Tensor, "batch layer pos"])
            Interpret pos_deltas as the position you want to virtually add
            Okay to modify in-place
        dependent_start : [batch] | int
    Returns:
        intervened_logits : [batch, pos, vocab_size]
    """
    pos_deltas = torch.zeros(
        (source_tokens.shape[0], model.cfg.n_layers, source_tokens.shape[1]),
        dtype=torch.int64,
    )
    if isinstance(dependent_start, int):
        dependent_start = torch.tensor([dependent_start] * source_tokens.shape[0])
    assert dependent_start.shape[0] == source_tokens.shape[0]
    assert len(dependent_start.shape) == 1

    if pos_deltas_fn is not None:
        pos_deltas = pos_deltas_fn(pos_deltas)

    target_dependent_mask = torch.zeros(
        (source_tokens.shape[0], model.cfg.n_layers, source_tokens.shape[1]), dtype=bool
    )
    for batch in range(source_tokens.shape[0]):
        target_dependent_mask[batch, :, dependent_start[batch] :] = True
    source_mask = ~target_dependent_mask  # batch, layer, pos

    source_logits, source_cache = model.run_with_cache(
        source_tokens, names_filter=lambda x: any(y in x for y in ["resid_pre"])
    )
    all_hooks = []
    for layer in range(model.cfg.n_layers):
        hook_fns = [
            partial(
                patch_masked_residue,
                mask=source_mask[:, layer, :],
                source_cache=source_cache,
            )
        ]

        hook_fns.append(hook_fn_generator(parameters, layer))

        all_hooks.append(
            (utils.get_act_name("resid_pre", layer), compose_hook_fns(hook_fns))
        )
        if pos_deltas_fn is not None:
            all_hooks.append(
                (
                    utils.get_act_name("rot_k", layer),
                    partial(
                        patch_rotary_k,
                        pos_deltas=pos_deltas[:, layer, :],
                        rotate_function=partial(
                            rotary_deltas, attn=model.blocks[layer].attn
                        ),
                    ),
                )
            )
    intervened_logits = model.run_with_hooks(
        source_tokens, fwd_hooks=all_hooks, return_type="logits"
    )
    return intervened_logits


def truncate_logits(logits, eps=1e-2):
    return logits
    return np.log(eps) + F.elu(logits - np.log(eps))


def train_multi(
    *,
    model,
    parameters,
    vocab,
    num_steps,
    batch_size=50,
    lr,
    category,
    reg_params=dict(),
    width,
    num_entities=2,
    output_steps=10,
    early_stop=None,
    project=False,
):
    """
    category : "names" | "countries" - this is the selector category
    """
    assert category == "names" or category == "countries"
    get_expected_answer = (
        lambda query_name, switch: switch[0] ^ switch[1] ^ query_name
        if query_name in switch
        else query_name
    )
    if not project:
        raise NotImplementedError
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    losses = []
    mags = []
    optimizer = SGD(parameters.get_param_list(), lr=lr, weight_decay=0.0)
    dependent_start = selector["dependent_start"]
    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    for step in range(num_steps):
        cur_losses = []
        for qn_subject in range(num_entities):
            for switch in get_pairwise_switches(num_entities):
                (
                    source_tokens,
                    answer_tokens,
                    query_names,
                    token_maps,
                ) = build_example_refactored(
                    batch_size=batch_size,
                    vocab=vocab,
                    fixed_query_name=qn_subject,
                    num_entities=num_entities,
                )
                intervened_logits = run_intervened_model(
                    model=model,
                    parameters=parameters,
                    source_tokens=source_tokens,
                    hook_fn_generator=apply_param_adapter(
                        hook_fn=partial(
                            project_hook_fn,
                            width=width,
                            category=category,
                            selector=selector,
                            coefficients=get_coefficients_from_switch(
                                num_entities, switch
                            ),
                        ),
                        param_adapter=parameters.default_adapter,
                    ),
                    dependent_start=dependent_start,
                )
                answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
                answer_logits = truncate_logits(answer_logits)
                expected_answer = get_expected_answer(qn_subject, switch)
                cur_loss = -answer_logits[:, expected_answer].mean()
                cur_loss.backward()
                cur_losses.append(cur_loss.item())

        weight_loss = parameters.weight_loss(parameters, **reg_params)
        weight_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss = sum(cur_losses) + weight_loss.item()
        losses.append(cur_losses)
        mags.append(parameters.weight_loss(parameters, weight=1.0).item())
        if step % output_steps == 0:
            print(
                f"{step:4d} {total_loss:+.3f} "
                + " ".join(f"{x:.3f}" for x in cur_losses)
                + f" {mags[-1]:.3f}"
            )
        if (
            early_stop is not None
            and step > 10
            and all(losses[-i] < early_stop for i in range(1, 10))
        ):
            break
    return losses, mags


def train_countries(
    *,
    model,
    parameters,
    vocab,
    num_steps,
    batch_size=50,
    lr,
    reg_params=dict(),
    num_entities=2,
    output_steps=10,
    early_stop=None,
    project=False,
):
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    losses = []
    mags = []
    optimizer = SGD(parameters.get_param_list(), lr=lr, weight_decay=0.0)
    dependent_start = selector["dependent_start"]
    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    for step in range(num_steps):
        source_tokens, answer_tokens = build_example(
            batch_size=batch_size, vocab=vocab, num_entities=num_entities
        )
        intervened_logits = run_intervened_model(
            model=model,
            parameters=parameters,
            source_tokens=source_tokens,
            hook_fn_generator=apply_param_adapter(
                hook_fn=partial(
                    project_countries_hook_fn,
                    selector=selector,
                    coefficients=[[-1, 1], [1, -1]],
                )
                if project
                else partial(inject_countries_hook_fn, selector=selector),
                param_adapter=parameters.default_adapter,
            ),
            dependent_start=dependent_start,
        )
        answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
        answer_logits = truncate_logits(answer_logits)

        logit_loss = torch.mean(-answer_logits[:, 1])
        loss = logit_loss + parameters.weight_loss(parameters, **reg_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        mags.append(parameters.weight_loss(parameters, weight=1.0).item())
        if step % output_steps == 0:
            print(
                f"{step:4d} {losses[-1]:+.3f} {logit_loss.item():+.3f} {mags[-1]:.3f}"
            )
        if (
            early_stop is not None
            and step > 10
            and all(losses[-i] < early_stop for i in range(1, 10))
        ):
            break
    return losses, mags


def train_entities(
    *,
    model,
    parameters,
    vocab,
    num_steps,
    batch_size=50,
    lr,
    reg_params=dict(),
    num_entities=2,
    output_steps=10,
    early_stop=None,
    project=False,
):
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    losses = []
    mags = []
    optimizer = SGD(parameters.get_param_list(), lr=lr, weight_decay=0.0)
    dependent_start = selector["dependent_start"]
    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    for step in range(num_steps):
        source_tokens, answer_tokens = build_example(
            batch_size=batch_size, vocab=vocab, num_entities=num_entities
        )
        intervened_logits = run_intervened_model(
            model=model,
            parameters=parameters,
            source_tokens=source_tokens,
            hook_fn_generator=apply_param_adapter(
                hook_fn=partial(
                    project_names_hook_fn,
                    selector=selector,
                    coefficients=[[-1, 1], [1, -1]],
                )
                if project
                else partial(inject_names_hook_fn, selector=selector),
                param_adapter=parameters.default_adapter,
            ),
            dependent_start=dependent_start,
        )
        answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
        answer_logits = truncate_logits(answer_logits)

        logit_loss = torch.mean(-answer_logits[:, 1])
        loss = logit_loss + parameters.weight_loss(parameters, **reg_params)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        mags.append(parameters.weight_loss(parameters, weight=1.0).item())
        if step % output_steps == 0:
            print(
                f"{step:4d} {losses[-1]:+.3f} {logit_loss.item():+.3f} {mags[-1]:.3f}"
            )
        if (
            early_stop is not None
            and step > 10
            and all(losses[-i] < early_stop for i in range(1, 10))
        ):
            break
    return losses, mags


def train_joint(
    *,
    model,
    entity_params,
    country_params,
    vocab,
    num_steps,
    batch_size=50,
    lr,
    reg_params=dict(),
    num_entities=2,
    output_steps=10,
    early_stop=None,
    one_sided=True,
):
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    losses = []
    mags = []
    compound_params = CompoundParameters(
        model=model, d={"country": country_params, "entity": entity_params}
    )
    optimizer = SGD(compound_params.get_param_list(), lr=lr, weight_decay=0.0)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=3000
    )
    dependent_start = selector["dependent_start"]
    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    for step in range(num_steps):
        source_tokens, answer_tokens = build_example(
            batch_size=batch_size, vocab=vocab, num_entities=num_entities
        )

        @lambda x: x()
        def country_loss():
            intervened_logits = run_intervened_model(
                model=model,
                parameters=country_params,
                source_tokens=source_tokens,
                hook_fn_generator=apply_param_adapter(
                    hook_fn=partial(inject_countries_hook_fn, selector=selector),
                    param_adapter=country_params.default_adapter,
                ),
                dependent_start=dependent_start,
            )
            answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
            answer_logits = truncate_logits(answer_logits)

            if one_sided:
                logit_loss = torch.mean(-answer_logits[:, 1])
            else:
                logit_loss = torch.mean(answer_logits[:, 0] - answer_logits[:, 1])
            return logit_loss

        @lambda x: x()
        def entity_loss():
            intervened_logits = run_intervened_model(
                model=model,
                parameters=entity_params,
                source_tokens=source_tokens,
                hook_fn_generator=apply_param_adapter(
                    hook_fn=partial(inject_names_hook_fn, selector=selector),
                    param_adapter=entity_params.default_adapter,
                ),
                dependent_start=dependent_start,
            )
            answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
            answer_logits = truncate_logits(answer_logits)

            if one_sided:
                logit_loss = torch.mean(-answer_logits[:, 1])
            else:
                logit_loss = torch.mean(answer_logits[:, 0] - answer_logits[:, 1])
            return logit_loss

        @lambda x: x()
        def joint_loss():
            intervened_logits = run_intervened_model(
                model=model,
                parameters=entity_params,
                source_tokens=source_tokens,
                hook_fn_generator=compose_hook_fn_generators(
                    [
                        apply_param_adapter(
                            partial(inject_names_hook_fn, selector=selector),
                            partial(
                                compound_params.default_adapter,
                                tag="entity",
                                child_adapter=entity_params.default_adapter,
                            ),
                        ),
                        apply_param_adapter(
                            partial(inject_countries_hook_fn, selector=selector),
                            partial(
                                compound_params.default_adapter,
                                tag="country",
                                child_adapter=country_params.default_adapter,
                            ),
                        ),
                    ]
                ),
                dependent_start=dependent_start,
            )
            answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
            answer_logits = truncate_logits(answer_logits)

            if one_sided:
                logit_loss = torch.mean(answer_logits[:, 0])
            else:
                logit_loss = torch.mean(answer_logits[:, 0] - answer_logits[:, 1])
            return logit_loss

        country_loss.backward()
        entity_loss.backward()
        loss = -joint_loss + compound_params.weight_loss(compound_params, **reg_params)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss += country_loss + entity_loss - joint_loss
        losses.append(loss.item())
        mags.append(compound_params.weight_loss(compound_params, weight=1.0).item())
        if step % output_steps == 0:
            print(
                f"{step:4d} {losses[-1]:+.3f} {country_loss.item():+.3f} {entity_loss.item():+.3f} {joint_loss.item():+.3f} {mags[-1]:.3f}"
            )
        if (
            early_stop is not None
            and step > 10
            and all(losses[-i] < early_stop for i in range(1, 10))
        ):
            break
    return losses, mags


def scale_params(params, factor, detach=True, acc_grad=False):
    new_params = copy.copy(params)  # shallow copy
    new_params.parameters = {}
    for k, v in params.parameters.items():
        new_params.parameters[k] = v * factor
        if detach:
            new_params.parameters[k] = new_params.parameters[k].detach()
            new_params.parameters[k].requires_grad = acc_grad

    return new_params


def add_params(params1, params2, detach=True, acc_grad=False):
    new_params = copy.copy(params1)  # shallow copy
    new_params.parameters = {}
    for k, v in params1.parameters.items():
        new_params.parameters[k] = v + params2.parameters[k]
        if detach:
            new_params.parameters[k] = new_params.parameters[k].detach()
            new_params.parameters[k].requires_grad = acc_grad

    return new_params


def concatenate_params(params, dim=0):
    final_params = copy.copy(params[0])
    final_params.parameters = {}
    for k in params[0].parameters.keys():
        final_params.parameters[k] = torch.concatenate(
            [param.parameters[k] for param in params], dim=dim
        ).to(params[0].parameters[k].device)
    return final_params


def switch_to_device(param, device):
    for k, v in param.parameters.items():
        param.parameters[k] = v.to(device)


def evaluate_subspace(
    *,
    selector,
    side,
    switch,
    model,
    source_tokens,
    answer_tokens,
    compound_params,
    country_params,
    entity_params,
    country_subspace,
    entity_subspace,
    project,
):
    intervened_logits = run_intervened_model(
        model=model,
        parameters=compound_params,
        source_tokens=source_tokens,
        hook_fn_generator=compose_hook_fn_generators(
            [
                apply_param_adapter(
                    hook_fn=partial(
                        project_names_hook_fn,
                        selector=selector,
                        coefficients=[[-1, 0], [0, -1]],
                    )
                    if project
                    else partial(inject_names_hook_fn, selector=selector),
                    param_adapter=partial(
                        compound_params.default_adapter,
                        tag="entity",
                        child_adapter=entity_params.default_adapter,
                    ),
                ),
                apply_param_adapter(
                    hook_fn=partial(
                        project_countries_hook_fn,
                        selector=selector,
                        coefficients=[[-1, 0], [0, -1]],
                    )
                    if project
                    else partial(inject_countries_hook_fn, selector=selector),
                    param_adapter=partial(
                        compound_params.default_adapter,
                        tag="country",
                        child_adapter=country_params.default_adapter,
                    ),
                ),
                apply_param_adapter(
                    partial(inject_names_hook_fn, selector=selector, side=side),
                    partial(
                        compound_params.default_adapter,
                        tag="entity_subspace",
                        child_adapter=entity_subspace.default_adapter,
                    ),
                ),
                apply_param_adapter(
                    partial(
                        inject_countries_hook_fn,
                        selector=selector,
                        side=side ^ switch,
                    ),
                    partial(
                        compound_params.default_adapter,
                        tag="country_subspace",
                        child_adapter=country_subspace.default_adapter,
                    ),
                ),
            ]
        ),
        dependent_start=selector["dependent_start"],
    )
    answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
    answer_logits = truncate_logits(answer_logits)
    return torch.mean(-answer_logits[:, switch])


def train_subspace(
    *,
    model,
    entity_params,
    country_params,
    entity_subspace,
    country_subspace,
    vocab,
    num_steps,
    batch_size=50,
    lr,
    reg_params=dict(),
    num_entities=2,
    output_steps=10,
    early_stop=None,
    project=False,
    gradient_projector=None,
):
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    losses = []
    mags = []
    compound_params = CompoundParameters(
        model=model,
        d={
            "country": scale_params(country_params, 0.5),
            "entity": scale_params(entity_params, 0.5),
            "country_subspace": country_subspace,
            "entity_subspace": entity_subspace,
        },
    )
    optimizer = SGD(
        country_subspace.get_param_list() + entity_subspace.get_param_list(),
        lr=lr,
        weight_decay=0.0,
    )
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=3000
    )
    dependent_start = selector["dependent_start"]
    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    for step in range(num_steps):
        source_tokens, answer_tokens = build_example(
            batch_size=batch_size,
            vocab=vocab,
            num_entities=num_entities,
        )

        def confusion_loss(side, switch):
            intervened_logits = run_intervened_model(
                model=model,
                parameters=compound_params,
                source_tokens=source_tokens,
                hook_fn_generator=compose_hook_fn_generators(
                    [
                        apply_param_adapter(
                            hook_fn=partial(
                                project_names_hook_fn,
                                selector=selector,
                                coefficients=[[-1, 0], [0, -1]],
                            )
                            if project
                            else partial(inject_names_hook_fn, selector=selector),
                            param_adapter=partial(
                                compound_params.default_adapter,
                                tag="entity",
                                child_adapter=entity_params.default_adapter,
                            ),
                        ),
                        apply_param_adapter(
                            hook_fn=partial(
                                project_countries_hook_fn,
                                selector=selector,
                                coefficients=[[-1, 0], [0, -1]],
                            )
                            if project
                            else partial(inject_countries_hook_fn, selector=selector),
                            param_adapter=partial(
                                compound_params.default_adapter,
                                tag="country",
                                child_adapter=country_params.default_adapter,
                            ),
                        ),
                        apply_param_adapter(
                            partial(inject_names_hook_fn, selector=selector, side=side),
                            partial(
                                compound_params.default_adapter,
                                tag="entity_subspace",
                                child_adapter=entity_subspace.default_adapter,
                            ),
                        ),
                        apply_param_adapter(
                            partial(
                                inject_countries_hook_fn,
                                selector=selector,
                                side=side ^ switch,
                            ),
                            partial(
                                compound_params.default_adapter,
                                tag="country_subspace",
                                child_adapter=country_subspace.default_adapter,
                            ),
                        ),
                    ]
                ),
                dependent_start=dependent_start,
            )
            answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
            answer_logits = truncate_logits(answer_logits)
            return torch.mean(-answer_logits[:, switch])

        confusion_00 = confusion_loss(0, 0)
        confusion_00.backward()
        confusion_10 = confusion_loss(1, 0)
        confusion_10.backward()
        confusion_01 = confusion_loss(0, 1)
        confusion_01.backward()
        confusion_11 = confusion_loss(1, 1)
        confusion_11.backward()
        loss = compound_params.weight_loss(compound_params, **reg_params)
        loss.backward()
        if gradient_projector is not None:
            gradient_projector(country_subspace, entity_subspace)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss += confusion_00 + confusion_10 + confusion_01 + confusion_11
        losses.append(loss.item())
        mags.append(compound_params.weight_loss(compound_params, weight=1.0).item())
        if step % output_steps == 0:
            print(
                f"{step:4d} {losses[-1]:+.3f} {confusion_00.item():+.3f} {confusion_10.item():+.3f} {confusion_01.item():+.3f} {confusion_11.item():+.3f} {mags[-1]:.3f}"
            )
        if (
            early_stop is not None
            and step > 10
            and all(losses[-i] < early_stop for i in range(1, 10))
        ):
            break
    return losses, mags


def get_hook_fns_for_subspace(
    selector,
    compound_params,
    country_params,
    entity_params,
    entity_subspace,
    country_subspace,
    base,
    side,
    switch,
):
    return compose_hook_fn_generators(
        [
            apply_param_adapter(
                hook_fn=partial(
                    project_names_hook_fn,
                    selector=selector,
                    coefficients=[[0, 0], [1, -1]] if base else [[-1, 1], [0, 0]],
                ),
                param_adapter=partial(
                    compound_params.default_adapter,
                    tag="entity",
                    child_adapter=entity_params.default_adapter,
                ),
            ),
            apply_param_adapter(
                hook_fn=partial(
                    project_countries_hook_fn,
                    selector=selector,
                    coefficients=[[0, 0], [1, -1]] if base else [[-1, 1], [0, 0]],
                ),
                param_adapter=partial(
                    compound_params.default_adapter,
                    tag="country",
                    child_adapter=country_params.default_adapter,
                ),
            ),
            apply_param_adapter(
                partial(inject_names_hook_fn, selector=selector, side=side),
                partial(
                    compound_params.default_adapter,
                    tag="entity_subspace",
                    child_adapter=entity_subspace.default_adapter,
                ),
            ),
            apply_param_adapter(
                partial(
                    inject_countries_hook_fn,
                    selector=selector,
                    side=side ^ switch,
                ),
                partial(
                    compound_params.default_adapter,
                    tag="country_subspace",
                    child_adapter=country_subspace.default_adapter,
                ),
            ),
        ]
    )


def train_subspace_swap_base(
    *,
    model,
    entity_params,
    country_params,
    entity_subspace,
    country_subspace,
    vocab,
    num_steps,
    batch_size=50,
    lr,
    reg_params=dict(),
    project=False,
    num_entities=2,
    output_steps=10,
    early_stop=None,
    gradient_projector=None,
):
    assert project
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    losses = []
    mags = []
    compound_params = CompoundParameters(
        model=model,
        d={
            "country": country_params,
            "entity": entity_params,
            "country_subspace": country_subspace,
            "entity_subspace": entity_subspace,
        },
    )
    optimizer = SGD(
        country_subspace.get_param_list() + entity_subspace.get_param_list(),
        lr=lr,
        weight_decay=0.0,
    )
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=3000
    )
    dependent_start = selector["dependent_start"]
    torch.set_grad_enabled(True)
    model.requires_grad_(False)

    for step in range(num_steps):
        source_tokens, answer_tokens = build_example(
            batch_size=batch_size,
            vocab=vocab,
            num_entities=num_entities,
        )

        def confusion_loss(side, switch):
            intervened_logits = run_intervened_model(
                model=model,
                parameters=compound_params,
                source_tokens=source_tokens,
                hook_fn_generator=get_hook_fns_for_subspace(
                    selector=selector,
                    compound_params=compound_params,
                    entity_params=entity_params,
                    country_params=country_params,
                    entity_subspace=entity_subspace,
                    country_subspace=country_subspace,
                    base=1,
                    side=side,
                    switch=switch,
                ),
                dependent_start=dependent_start,
            )
            answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)
            answer_logits = truncate_logits(answer_logits)
            return torch.mean(-answer_logits[:, switch])

        confusion_00 = confusion_loss(0, 0)
        confusion_00.backward()
        confusion_10 = confusion_loss(1, 0)
        confusion_10.backward()
        confusion_01 = confusion_loss(0, 1)
        confusion_01.backward()
        confusion_11 = confusion_loss(1, 1)
        confusion_11.backward()
        loss = compound_params.weight_loss(compound_params, **reg_params)
        loss.backward()
        if gradient_projector is not None:
            gradient_projector(country_subspace, entity_subspace)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        loss += confusion_00 + confusion_10 + confusion_01 + confusion_11
        losses.append(loss.item())
        mags.append(compound_params.weight_loss(compound_params, weight=1.0).item())
        if step % output_steps == 0:
            print(
                f"{step:4d} {losses[-1]:+.3f} {confusion_00.item():+.3f} {confusion_10.item():+.3f} {confusion_01.item():+.3f} {confusion_11.item():+.3f} {mags[-1]:.3f}"
            )
        if (
            early_stop is not None
            and step > 10
            and all(losses[-i] < early_stop for i in range(1, 10))
        ):
            break
    return losses, mags


def evaluate_params(
    model,
    parameters,
    num_samples,
    vocab,
    hook_fn_generator,
    pos_deltas_fn=None,
    batch_size=50,
    fixed_query_name=None,
    num_entities=2,
):
    loss_sum = 0
    logprob_sum = 0
    selector = build_selector(vocab=vocab, num_entities=2)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        source_tokens, answer_tokens = build_example(
            batch_size=end - start,
            vocab=vocab,
            fixed_query_name=fixed_query_name,
            num_entities=num_entities,
        )
        intervened_logits = run_intervened_model(
            model=model,
            parameters=parameters,
            source_tokens=source_tokens,
            hook_fn_generator=hook_fn_generator,
            dependent_start=selector["dependent_start"],
            pos_deltas_fn=pos_deltas_fn,
        )
        answer_logits = lookup_answer_logits(intervened_logits, answer_tokens)

        loss = torch.sum(answer_logits[:, 0] - answer_logits[:, 1])
        loss_sum += loss
        logprob_sum += answer_logits[:, 1].sum()
    return loss_sum / num_samples, logprob_sum / num_samples


def evaluate_params_refactored(
    model,
    parameters,
    num_samples,
    vocab,
    hook_fn_generator,
    pos_deltas_fn=None,
    batch_size=50,
    fixed_query_name=None,
    raw_query_name=None,
    num_entities=2,
    num_answers=None,
    additional_answer_tokens=[],
    prompt_id_start=None,
    return_raw=False,
    template_type="normal",
    context=None,
    template_context=None,
    normalize_logits=True,
):
    """
    Args:
        fixed_query_name : None | Name
        raw_query_name : None | str
    """
    if template_context is not None:
        assert fixed_query_name is None and raw_query_name is None
    answer_logits_sum = torch.zeros(num_entities + len(additional_answer_tokens))
    all_answer_logits = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        (
            source_tokens,
            answer_tokens,
            query_names,
            token_maps,
        ) = build_example_refactored(
            batch_size=end - start,
            vocab=vocab,
            fixed_query_name=fixed_query_name,
            raw_query_name=raw_query_name,
            num_entities=num_entities,
            prompt_id_start=None
            if prompt_id_start is None
            else prompt_id_start + start,
            template_type=template_type,
            template_context=template_context,
            context=context,
            num_answers=num_answers,
        )

        # print(model.to_string(source_tokens)[0])
        # print(token_maps)
        # print(model.to_string(answer_tokens))
        # for batch in range(source_tokens.shape[0]):
        #     print(f'batch {batch}')
        #     print(', '.join(f'({vocab.tokenizer.decode(token)} {pos})' for pos, token in enumerate(source_tokens[batch])))
        if prompt_id_start is None:
            prompt_ids = None
        else:
            prompt_ids = slice(prompt_id_start + start, prompt_id_start + end)
        intervened_logits = run_intervened_model(
            model=model,
            parameters=parameters,
            source_tokens=source_tokens,
            hook_fn_generator=partial(
                hook_fn_generator, prompt_ids=prompt_ids, token_maps=token_maps
            ),
            dependent_start=token_maps["context_section"][:, 1],
            pos_deltas_fn=None
            if pos_deltas_fn is None
            else partial(pos_deltas_fn, token_maps=token_maps),
        )
        if (
            fixed_query_name is None
            and raw_query_name is None
            and template_context is None
        ):
            reorder_answers(answer_tokens, query_names)
            print("Warning: reordering")
            assert False
        if additional_answer_tokens:
            additional_answer_tokens_tensor = torch.tensor(additional_answer_tokens).to(
                answer_tokens.device
            )
            answer_tokens = torch.cat(
                [
                    answer_tokens,
                    repeat(
                        additional_answer_tokens_tensor,
                        "d-> batch d",
                        batch=answer_tokens.shape[0],
                    ),
                ],
                dim=1,
            )
        answer_logits = lookup_answer_logits(
            intervened_logits,
            answer_tokens,
            query_position=token_maps["prompt"][:, 1] - 1,
            normalize=normalize_logits,
        )
        if not return_raw:
            answer_logits_sum += answer_logits.sum(dim=0).detach().cpu()
        else:
            all_answer_logits.append(answer_logits.detach().cpu())
    if not return_raw:
        return answer_logits_sum / num_samples
    else:
        return torch.cat(all_answer_logits, dim=0)


def get_pairwise_switches(num_entities):
    return [(i, j) for i in range(num_entities) for j in range(num_entities) if i < j]


def get_cyclic_perm(num_entities):
    assert num_entities == 3
    l = list(range(num_entities))
    return [l[1:] + l[:1], l[-1:] + l[:-1]]


def get_coefficients_from_switch(num_entities, switch):
    coefficients = [[0 for i in range(num_entities)] for j in range(num_entities)]
    assert switch[0] != switch[1]
    coefficients[switch[0]][switch[1]] = 1
    coefficients[switch[1]][switch[0]] = 1
    coefficients[switch[0]][switch[0]] = -1
    coefficients[switch[1]][switch[1]] = -1
    return coefficients


def validate_country_switch(
    model,
    vocab,
    country_params,
    num_samples=50,
    project=False,
    num_entities=2,
    get_expected_answer=lambda query_name, switch: switch[0] ^ switch[1] ^ query_name
    if query_name in switch
    else query_name,
    verbose=False,
    raw_logits=False,
    return_type=None,
    promote_all=False,
    demote_all=False,
):
    """
    We're deprecating raw_logits.
    return_type is now mandatory.
    raw_logits = True -> return_type = "logits"
    raw_logits = False -> return_type = "difference"

    Args:
        raw_logits: bool - if True, return logit array instead of difference statistic
        return_type: None | "logits" | "difference" | "raw"
    """
    if return_type is None:
        warnings.warn(
            "return_type is None. raw_logits has be deprecated. Please specify return_type.",
            DeprecationWarning,
        )
        return_type = "logits" if raw_logits else "difference"
    gg = []
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    if promote_all:

        def pos_deltas_fn(pos_deltas):
            for p in selector["countries"]:
                pos_deltas[:, :, p] += selector["countries"][-1] - p
            return pos_deltas

    elif demote_all:

        def pos_deltas_fn(pos_deltas):
            for p in selector["countries"]:
                pos_deltas[:, :, p] += selector["countries"][0] - p
            return pos_deltas

    else:
        pos_deltas_fn = None
    for qn_subject in range(num_entities):
        cur = []
        for switch in get_pairwise_switches(num_entities):
            answer_logprobs = evaluate_params_refactored(
                model=model,
                parameters=country_params,
                num_samples=num_samples,
                vocab=vocab,
                num_entities=num_entities,
                hook_fn_generator=compose_hook_fn_generators(
                    [
                        apply_param_adapter(
                            hook_fn=partial(
                                project_countries_hook_fn,
                                selector=selector,
                                coefficients=get_coefficients_from_switch(
                                    num_entities, switch
                                ),
                            )
                            if project
                            else partial(inject_countries_hook_fn, selector=selector),
                            param_adapter=partial(country_params.default_adapter),
                        ),
                    ]
                ),
                fixed_query_name=qn_subject,
                pos_deltas_fn=pos_deltas_fn,
                return_raw=(return_type == "raw"),
            )
            expected_answer = get_expected_answer(qn_subject, switch)
            if verbose:
                print(
                    f"{qn_subject} {switch} {expected_answer} {get_coefficients_from_switch(num_entities, switch)}"
                )
            if return_type == "logits" or return_type == "raw":
                cur.append(answer_logprobs.detach().cpu().numpy())
            else:
                cur.append(
                    [
                        answer_logprobs[expected_answer].item(),
                        (
                            answer_logprobs.sum() - answer_logprobs[expected_answer]
                        ).item()
                        / (num_entities - 1),
                    ]
                )
        gg.append(cur)
    return np.array(gg)


def validate_entity_switch(
    model,
    vocab,
    entity_params,
    width=2,
    num_samples=50,
    project=False,
    num_entities=2,
    get_expected_answer=lambda query_name, switch: switch[0] ^ switch[1] ^ query_name
    if query_name in switch
    else query_name,
    verbose=False,
    raw_logits=False,
    promote_all=False,
    demote_all=False,
):
    gg = []
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    if promote_all:

        def pos_deltas_fn(pos_deltas):
            for p in selector["names"]:
                pos_deltas[:, :, p : p + width] += selector["names"][-1] - p
            return pos_deltas

    elif demote_all:

        def pos_deltas_fn(pos_deltas):
            for p in selector["names"]:
                pos_deltas[:, :, p : p + width] += selector["names"][0] - p
            return pos_deltas

    else:
        pos_deltas_fn = None
    for qn_subject in range(num_entities):
        cur = []
        for switch in get_pairwise_switches(num_entities):
            answer_logprobs = evaluate_params_refactored(
                model=model,
                parameters=entity_params,
                num_samples=num_samples,
                vocab=vocab,
                num_entities=num_entities,
                hook_fn_generator=compose_hook_fn_generators(
                    [
                        apply_param_adapter(
                            hook_fn=partial(
                                project_names_hook_fn,
                                width=width,
                                selector=selector,
                                coefficients=get_coefficients_from_switch(
                                    num_entities, switch
                                ),
                            )
                            if project
                            else partial(
                                inject_names_hook_fn, selector=selector, width=width
                            ),
                            param_adapter=partial(entity_params.default_adapter),
                        ),
                    ]
                ),
                fixed_query_name=qn_subject,
                pos_deltas_fn=pos_deltas_fn,
            )
            expected_answer = get_expected_answer(qn_subject, switch)
            if verbose:
                print(
                    f"{qn_subject} {switch} {expected_answer} {get_coefficients_from_switch(num_entities, switch)}"
                )
            if raw_logits:
                cur.append(answer_logprobs.detach().cpu().numpy())
            else:
                cur.append(
                    [
                        answer_logprobs[expected_answer].item(),
                        (
                            answer_logprobs.sum() - answer_logprobs[expected_answer]
                        ).item()
                        / (num_entities - 1),
                    ]
                )
        gg.append(cur)
    return np.array(gg)


def validate_both_switch(
    model,
    vocab,
    entity_params,
    country_params,
    project=False,
    num_entities=2,
    get_expected_answer=lambda query_name, switch: query_name
    if query_name in switch
    else query_name,
    verbose=False,
    raw_logits=False,
):
    gg = []

    selector = build_selector(vocab=vocab, num_entities=num_entities)

    compound_params = CompoundParameters(
        model=model, d={"country": country_params, "entity": entity_params}
    )
    for qn_subject in range(num_entities):
        cur = []
        for switch in get_pairwise_switches(num_entities):
            answer_logprobs = evaluate_params_refactored(
                model=model,
                parameters=compound_params,
                num_samples=50,
                vocab=vocab,
                num_entities=num_entities,
                hook_fn_generator=compose_hook_fn_generators(
                    [
                        apply_param_adapter(
                            hook_fn=partial(
                                project_names_hook_fn,
                                selector=selector,
                                coefficients=get_coefficients_from_switch(
                                    num_entities, switch
                                ),
                            )
                            if project
                            else partial(inject_names_hook_fn, selector=selector),
                            param_adapter=partial(
                                compound_params.default_adapter,
                                tag="entity",
                                child_adapter=entity_params.default_adapter,
                            ),
                        ),
                        apply_param_adapter(
                            hook_fn=partial(
                                project_countries_hook_fn,
                                selector=selector,
                                coefficients=get_coefficients_from_switch(
                                    num_entities, switch
                                ),
                            )
                            if project
                            else partial(inject_countries_hook_fn, selector=selector),
                            param_adapter=partial(
                                compound_params.default_adapter,
                                tag="country",
                                child_adapter=country_params.default_adapter,
                            ),
                        ),
                    ]
                ),
                fixed_query_name=qn_subject,
            )
            expected_answer = get_expected_answer(qn_subject, switch)
            if verbose:
                print(
                    f"{qn_subject} {switch} {expected_answer} {get_coefficients_from_switch(num_entities, switch)}"
                )
            if raw_logits:
                cur.append(answer_logprobs.detach().cpu().numpy())
            else:
                cur.append(
                    [
                        answer_logprobs[expected_answer].item(),
                        (
                            answer_logprobs.sum() - answer_logprobs[expected_answer]
                        ).item()
                        / (num_entities - 1),
                    ]
                )
        gg.append(cur)
    return np.array(gg)


def classify_country(
    model, vocab, side, param, width, new_answer_token=None, return_matrix=False
):
    """
    Given a country param, classify whether it's binded to entity = side
    Outputs a number that is roughly 8 when it's correct and 0 when it's wrong
    See data_science_decision_boundary.ipynb for usage
    """
    assert side == 0 or side == 1
    num_entities = 2
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    answer_matrix = []
    additional_answer_tokens = None if new_answer_token is None else [new_answer_token]
    for qn_subject in range(num_entities):
        logits = evaluate_params_refactored(
            model=model,
            parameters=param,
            num_samples=50,
            vocab=vocab,
            num_entities=num_entities,
            hook_fn_generator=compose_hook_fn_generators(
                [
                    apply_param_adapter(
                        hook_fn=partial(
                            overwrite_hook_fn,
                            selector=selector,
                            category="countries",
                            width=width,
                            side=side,
                        ),
                        param_adapter=partial(param.default_adapter),
                    ),
                ]
            ),
            fixed_query_name=qn_subject,
            additional_answer_tokens=additional_answer_tokens,
        )
        answer_matrix.append(logits.detach().cpu().numpy())
    answer_matrix = np.array(answer_matrix)
    if return_matrix:
        return answer_matrix
    else:
        answer_matrix[:, side] = answer_matrix[:, -1]
        # answer_matrix = np.exp(answer_matrix)
        return (
            answer_matrix[0, 0]
            + answer_matrix[1, 1]
            - answer_matrix[0, 1]
            - answer_matrix[1, 0]
        )


def sub_param(
    *,
    model,
    vocab,
    param_map,
    additional_raw_query_names=[],
    additional_answer_tokens,
    prompt_id_start,
    num_samples=50,
    mode="sub",
    num_query_entities=2,
    swap_countries=False,
    country_params=None,
):
    """
    Substitutes parameters for both sides into the specified category
    Args:
        param_map : {
            "names" | "countries" -> {
                "width" -> width: int,
                sides : {
                    0 | 1 -> {
                        "param" -> param
                        "rotate" -> "first" | "last" | None
                    }
                }
            }
        }
        mode: 'sub' | 'inject'
        swap_countries: If true, then must specify country_params, which will be used to swap
    """
    num_entities = 2
    selector = build_selector(vocab=vocab, num_entities=num_entities)
    answer_matrix = []

    def pos_deltas_fn(pos_deltas):
        for category, deets in param_map.items():
            width = deets["width"]
            for side, side_deets in deets["sides"].items():
                rotate = side_deets["rotate"]
                if rotate == "first":
                    target = selector[category][0]
                elif rotate == "last":
                    target = selector[category][-1]
                elif rotate is None:
                    target = selector[category][side]
                else:
                    raise ValueError(f"Unknown rotate {rotate}")
                pos_deltas[
                    :, :, selector[category][side] : selector[category][side] + width
                ] += (target - selector[category][side])
        return pos_deltas

    compound_param = CompoundParameters(
        model=model,
        d={
            **{
                (category, side): param_map[category]["sides"][side]["param"]
                for category in param_map
                for side in param_map[category]["sides"]
            },
            "pre": country_params,
        },
    )
    for i, qn_subject in enumerate(
        list(range(num_query_entities)) + additional_raw_query_names
    ):
        fixed_mode = i < num_query_entities
        logits = evaluate_params_refactored(
            model=model,
            parameters=compound_param,
            num_samples=num_samples,
            vocab=vocab,
            num_entities=num_entities,
            prompt_id_start=prompt_id_start,
            hook_fn_generator=compose_hook_fn_generators(
                (
                    [
                        apply_param_adapter(
                            hook_fn=partial(
                                inject_hook_fn,
                                selector=selector,
                                category="countries",
                                width=param_map["countries"]["width"],
                            ),
                            param_adapter=partial(
                                compound_param.default_adapter,
                                tag="pre",
                                child_adapter=compound_param.parameters[
                                    "pre"
                                ].default_adapter,
                            ),
                        )
                    ]
                    if swap_countries
                    else []
                )
                + [
                    apply_param_adapter(
                        hook_fn=partial(
                            overwrite_hook_fn if mode == "sub" else inject_hook_fn,
                            selector=selector,
                            category=category,
                            width=param_map[category]["width"],
                            side=side
                            if category != "countries" or not swap_countries
                            else 1 - side,
                        ),
                        param_adapter=partial(
                            compound_param.default_adapter,
                            tag=(category, side),
                            child_adapter=compound_param.parameters[
                                (category, side)
                            ].default_adapter,
                        ),
                    )
                    for category in param_map
                    for side in param_map[category]["sides"]
                ]
            ),
            additional_answer_tokens=additional_answer_tokens,
            pos_deltas_fn=pos_deltas_fn,
            **(
                dict(fixed_query_name=qn_subject)
                if fixed_mode
                else dict(raw_query_name=qn_subject)
            ),
        )
        answer_matrix.append(logits.detach().cpu().numpy())
    answer_matrix = np.array(answer_matrix)
    return answer_matrix
