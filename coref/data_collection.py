import gc
import numpy as np
from functools import partial
import torch
import einops
import coref.representations as rep
import coref.datascience as ds
from coref.experiments.common import repeat


def clear_garbage():
    gc.collect()
    torch.cuda.empty_cache()


def default_mcq_selector_extractor(token_maps):
    return {
        "options": [statement["option"][:, 0] for statement in token_maps["context"]],
        "labels": [statement["label"][:, 0] for statement in token_maps["context"]],
    }


def statements_mcq_selector_extractor(token_maps):
    return {
        "sentence_starts": [
            statement["sentence"][:, 0] for statement in token_maps["context"]
        ],
        "sentence_ends": [
            statement["sentence"][:, 1] for statement in token_maps["context"]
        ],
    }


def default_binding_selector_extractor(token_maps):
    def flatten(ls):
        return [a for b in ls for a in b]

    return {
        "names": flatten(
            [
                [statement["subject"][:, 0]]
                if isinstance(statement["subject"], torch.Tensor)
                else [k[:, 0] for k in statement["subject"]]
                for statement in token_maps["context"]
            ]
        ),
        "countries": flatten(
            [
                [statement["country"][:, 0]]
                if isinstance(statement["country"], torch.Tensor)
                else [k[:, 0] for k in statement["country"]]
                for statement in token_maps["context"]
            ]
        ),
    }


def compose_interventions(*, model, intervention_list):
    """
    Args:
        intervention_list: List[Tuple[Parameter, Param Adapter, Hook fn]]
    Returns:
        parameter, hook_fn_generator
    """
    compound_params = rep.CompoundParameters(
        model=model,
        d={idx: param for idx, (param, _, _) in enumerate(intervention_list)},
    )
    hook_fn_generator = rep.compose_hook_fn_generators(
        [
            rep.apply_param_adapter(
                hook_fn=hook_fn,
                param_adapter=partial(
                    rep.CompoundParameters.default_adapter,
                    tag=idx,
                    child_adapter=param_adapter,
                ),
            )
            for idx, (_, param_adapter, hook_fn) in enumerate(intervention_list)
        ]
    )
    return compound_params, hook_fn_generator


def factorization_parallel_exchange(
    *,
    model,
    vocab,
    num_entities,
    prompt_id_start,
    widths,
    num_samples=100,
    batch_size=50,
    template_type="normal",
    additional_template_context=None,
    base_selector_extractor,
):
    assert num_entities == 2
    # selector = rep.build_selector(vocab, num_entities)
    def data_selector_extractor(token_maps):
        sel = base_selector_extractor(token_maps)
        return {
            key: [[positions + i for i in range(width)] for positions in sel[key]]
            for key, width in widths.items()
        }

    def evaluate(swap_first, swap_second):
        source_context = vocab.default_context(
            num_entities=2,
            entities=range(num_entities, 2 * num_entities),
            attributes=range(num_entities, 2 * num_entities),
        )
        contrast_reps, contrast_tokens, _ = ds.collect_data(
            model=model,
            selector=None,
            selector_extractor=data_selector_extractor,
            num_entities=num_entities,
            num_samples=num_samples,
            batch_size=batch_size,
            vocab=vocab,
            context=source_context,
            index_start=prompt_id_start,
            hook_name="resid_pre",
            template_type=template_type,
            template_context=dict(
                query_name=0, raw_query_name=None, **additional_template_context
            ),
        )
        for k, v in contrast_reps.items():
            for i in range(len(v)):
                v[i] = einops.rearrange(
                    torch.stack(v[i]), "width batch layer dim -> layer batch width dim"
                )
        intervention_list = []
        exchange_ids = []
        if swap_first:
            exchange_ids.append((0, 0))
        if swap_second:
            exchange_ids.append((1, 1))
        for source_id, target_id in exchange_ids:
            for int_type, width in widths.items():
                if width > 0:

                    @lambda x: partial(x, target_id=target_id, int_type=int_type)
                    def extractor(token_maps, int_type, target_id):
                        sel = base_selector_extractor(token_maps)
                        return sel[int_type][target_id]

                    param = rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: contrast_reps[int_type][source_id][layer]
                            for layer in range(model.cfg.n_layers)
                        },
                    )
                    intervention_list.append(
                        (
                            param,
                            partial(
                                param.batch_adapter, prompt_id_start=prompt_id_start
                            ),
                            partial(
                                rep.general_overwrite_hook_fn,
                                width=width,
                                extractor=extractor,
                            ),
                        )
                    )
        compound_params, hook_fn_generator = compose_interventions(
            model=model, intervention_list=intervention_list
        )
        return repeat(
            num_entities * 2,
            lambda query_name: rep.evaluate_params_refactored(
                model=model,
                parameters=compound_params,
                num_samples=num_samples,
                batch_size=batch_size,
                vocab=vocab,
                context=vocab.default_context(num_entities=num_entities),
                hook_fn_generator=hook_fn_generator,
                num_entities=num_entities,
                prompt_id_start=prompt_id_start,
                return_raw=True,
                template_type=template_type,
                template_context=dict(
                    query_name=query_name,
                    raw_query_name=None,
                    **additional_template_context,
                ),
                num_answers=2 * num_entities,
            ),
        )

    return repeat(
        2,
        lambda swap_first: repeat(
            2, lambda swap_second: evaluate(swap_first, swap_second)
        ),
    )  # swap_first, swap_second, name, batch, answer


def factorization_parallel_exchange_by_side(
    *,
    model,
    vocab,
    num_entities,
    prompt_id_start,
    widths,
    side,
    num_samples=100,
    batch_size=50,
    template_type="normal",
    additional_template_context=None,
    base_selector_extractor,
):
    assert num_entities == 2
    # selector = rep.build_selector(vocab, num_entities)
    def data_selector_extractor(token_maps):
        sel = base_selector_extractor(token_maps)
        return {
            key: [[positions + i for i in range(width)] for positions in sel[key]]
            for key, width in widths.items()
        }

    source_context = vocab.default_context(
        num_entities=2,
        entities=range(num_entities, 2 * num_entities),
        attributes=range(num_entities, 2 * num_entities),
    )
    contrast_reps, contrast_tokens, _ = ds.collect_data(
        model=model,
        selector=None,
        selector_extractor=data_selector_extractor,
        num_entities=num_entities,
        num_samples=num_samples,
        batch_size=batch_size,
        vocab=vocab,
        context=source_context,
        index_start=prompt_id_start,
        hook_name="resid_pre",
        template_type=template_type,
        template_context=dict(
            query_name=0, raw_query_name=None, **additional_template_context
        ),
    )
    for k, v in contrast_reps.items():
        for i in range(len(v)):
            v[i] = einops.rearrange(
                torch.stack(v[i]), "width batch layer dim -> layer batch width dim"
            )

    def evaluate(swap_entity, swap_attribute):
        intervention_list = []
        for int_type, width in widths.items():
            if width > 0:
                if (int_type == "countries" and swap_attribute) or (
                    int_type == "names" and swap_entity
                ):

                    @lambda x: partial(x, target_id=side, int_type=int_type)
                    def extractor(token_maps, int_type, target_id):
                        sel = base_selector_extractor(token_maps)
                        return sel[int_type][target_id]

                    param = rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: contrast_reps[int_type][side][layer]
                            for layer in range(model.cfg.n_layers)
                        },
                    )
                    intervention_list.append(
                        (
                            param,
                            partial(
                                param.batch_adapter, prompt_id_start=prompt_id_start
                            ),
                            partial(
                                rep.general_overwrite_hook_fn,
                                width=width,
                                extractor=extractor,
                            ),
                        )
                    )
        compound_params, hook_fn_generator = compose_interventions(
            model=model, intervention_list=intervention_list
        )
        return repeat(
            num_entities * 2,
            lambda query_name: rep.evaluate_params_refactored(
                model=model,
                parameters=compound_params,
                num_samples=num_samples,
                batch_size=batch_size,
                vocab=vocab,
                context=vocab.default_context(num_entities=num_entities),
                hook_fn_generator=hook_fn_generator,
                num_entities=num_entities,
                prompt_id_start=prompt_id_start,
                return_raw=True,
                template_type=template_type,
                template_context=dict(
                    query_name=query_name,
                    raw_query_name=None,
                    **additional_template_context,
                ),
                num_answers=2 * num_entities,
            ),
        )

    return repeat(
        2,
        lambda swap_entity: repeat(
            2, lambda swap_attribute: evaluate(swap_entity, swap_attribute)
        ),
    )  # swap_entity, swap_attribute, name, batch, answer


def build_selector_refactored(base_selector_extractor, vocab, num_entities, additional_template_context):
    tokens, answers, qry_names, token_maps = rep.build_example_refactored(
        batch_size=1,
        vocab=vocab,
        context=vocab.default_context(num_entities),
        num_entities=num_entities,
        template_context=dict(raw_query_name=None, query_name=0, **additional_template_context),
        prompt_id_start=0,
    )
    return base_selector_extractor(token_maps)


def position_intervention(
    *,
    model,
    vocab,
    category,
    width,
    num_entities,
    prompt_id_start,
    num_samples,
    template_type,
    additional_template_context,
    base_selector_extractor,
    normalize_logits,
    fixed_offset_start=None,
    fixed_offset_end=None,
    is_tail=False
):
    assert num_entities == 2

    def pos_deltas_fn(pos_deltas, token_maps, offset):
        selector = base_selector_extractor(token_maps)
        for p, coeff in zip(selector[category], [1, -1]):
            for batch, pos in enumerate(p):
                pos_deltas[batch, :, pos : pos + width] += offset * coeff
        return pos_deltas

    selector = build_selector_refactored(base_selector_extractor, vocab, num_entities, additional_template_context)

    token_diff = selector[category][1][0].item() - selector[category][0][0].item()
    offset_lower_range = -10
    offset_upper_range = (
        10 + token_diff
    )
    if fixed_offset_start is not None:
        if is_tail:
            offset_lower_range = token_diff + fixed_offset_start
        else:
            offset_lower_range = fixed_offset_start
    if fixed_offset_end is not None:
        if is_tail:
            offset_upper_range = token_diff + fixed_offset_end
        else:
            offset_upper_range = fixed_offset_end
    return repeat(
        offset_upper_range - offset_lower_range,
        lambda offset: repeat(
            num_entities,
            lambda query_name: rep.evaluate_params_refactored(
                model=model,
                context=vocab.default_context(num_entities=num_entities),
                parameters=rep.UntiedInjectionParameters.from_dict(model, {}),
                num_samples=num_samples,
                vocab=vocab,
                hook_fn_generator=rep.apply_param_adapter(
                    None, rep.UntiedInjectionParameters.default_adapter
                ),
                num_entities=num_entities,
                prompt_id_start=prompt_id_start,
                return_raw=True,
                pos_deltas_fn=partial(
                    pos_deltas_fn, offset=offset + offset_lower_range
                ),
                template_type=template_type,
                template_context=dict(
                    query_name=query_name,
                    raw_query_name=None,
                    **additional_template_context,
                ),
                normalize_logits=normalize_logits,
            ),
        ),
    )


def parallel_inject(
    *,
    model,
    vocab,
    param_list,
    num_samples=50,
    batch_size=50,
    num_entities,
    base_selector_extractor,
    promotions=[],
    demotions=[],
    template_type,
    context,
    additional_template_context,
    include_null=False,
    prompt_id_start=0,
):
    """
    promotions: List[Tuple[category, width]]
    """
    assert (not promotions) or (not demotions)
    gg = []
    if promotions:

        def pos_deltas_fn(pos_deltas, token_maps):
            selector = base_selector_extractor(token_maps)
            for category, width in promotions:
                for p in selector[category]:
                    for batch, pos in enumerate(p):
                        pos_deltas[batch, :, pos : pos + width] += (
                            selector[category][-1][batch] - pos
                        )
            return pos_deltas

    elif demotions:

        def pos_deltas_fn(pos_deltas, token_maps):
            selector = base_selector_extractor(token_maps)
            for category, width in demotions:
                for p in selector[category]:
                    for batch, pos in enumerate(p):
                        pos_deltas[batch, :, pos : pos + width] += (
                            selector[category][0][batch] - pos
                        )
            return pos_deltas

    else:
        pos_deltas_fn = None
    for qn_subject in range(num_entities + include_null):

        def build_extractor(category, side):
            def extractor(token_maps):
                selector = base_selector_extractor(token_maps)
                return selector[category][side]

            return extractor

        intervention_list = [
            (
                p["params"],
                p["params"].default_adapter,
                partial(
                    rep.general_inject_hook_fn,
                    extractor=build_extractor(p["category"], p["side"]),
                    width=p["width"],
                ),
            )
            for p in param_list
        ]
        compound_params, hook_fn_generator = compose_interventions(
            model=model, intervention_list=intervention_list
        )

        answer_logprobs = rep.evaluate_params_refactored(
            model=model,
            parameters=compound_params,
            num_samples=num_samples,
            context=context,
            batch_size=batch_size,
            vocab=vocab,
            num_entities=num_entities,
            hook_fn_generator=hook_fn_generator,
            prompt_id_start=prompt_id_start,
            pos_deltas_fn=pos_deltas_fn,
            return_raw=True,
            template_type=template_type,
            template_context=dict(
                query_name=qn_subject,
                raw_query_name=None,
                **additional_template_context,
            ),
        )
        gg.append(answer_logprobs.detach().cpu())
    return torch.stack(gg)  # [name, answer]


def test_mean_interventions(
    *,
    model,
    vocab,
    num_entities,
    means,
    include_null,
    base_selector_extractor,
    template_type,
    additional_template_context,
    num_samples,
    prompt_id_start=0,
):
    assert num_entities == 2

    def getstuff(swap_entities, swap_countries):
        valid_categories = []
        if swap_entities:
            valid_categories.append("names")
        if swap_countries:
            valid_categories.append("countries")
        param_list = [
            [
                {
                    "category": category,
                    "params": rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: means[category][0][:, layer, :]
                            for layer in range(model.cfg.n_layers)
                        },
                    ),
                    "side": 0,
                    "width": means[category][0].shape[0],
                },
                {
                    "category": category,
                    "params": rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: -means[category][0][:, layer, :]
                            for layer in range(model.cfg.n_layers)
                        },
                    ),
                    "side": 1,
                    "width": means[category][0].shape[0],
                },
            ]
            for category in valid_categories
        ]
        param_list = [a for b in param_list for a in b]  # flatten
        return parallel_inject(
            model=model,
            vocab=vocab,
            param_list=param_list,
            num_samples=num_samples,
            batch_size=50,
            num_entities=num_entities,
            context=vocab.default_context(num_entities=num_entities),
            base_selector_extractor=base_selector_extractor,
            promotions=[("countries", 1)] if swap_countries else [],
            demotions=[],
            template_type=template_type,
            additional_template_context=additional_template_context,
            include_null=include_null,
            prompt_id_start=prompt_id_start,
        )

    return repeat(
        2,
        lambda swap_entities: repeat(
            2, lambda swap_countries: getstuff(swap_entities, swap_countries)
        ),
    )  # swap_entities, swap_countries, names, batch, attr


def test_subspace(
    *,
    model,
    vocab,
    num_entities,
    base_means,
    param_means,
    coeffs,
    include_null,
    prompt_id_start=0,
    base_selector_extractor,
    template_type,
    additional_template_context,
    cross_countries,
    num_samples,
):
    assert num_entities == 2
    # first, subtract base
    param_list = [
        [
            {
                "category": category,
                "params": rep.UntiedInjectionParameters.from_dict(
                    model,
                    {
                        layer: -base_means[category][0][:, layer, :]
                        for layer in range(model.cfg.n_layers)
                    },
                ),
                "side": 1,
                "width": base_means[category][0].shape[0],
            }
        ]
        for category in ["names", "countries"]
    ]
    # then, add param
    param_list += [
        [
            {
                "category": category,
                "params": rep.UntiedInjectionParameters.from_dict(
                    model,
                    {
                        layer: coeffs[side][0] * param_means[category][0][:, layer, :]
                        + coeffs[side][1] * param_means[category][1][:, layer, :]
                        for layer in range(model.cfg.n_layers)
                    },
                ),
                "side": 1 - side
                if cross_countries and category == "countries"
                else side,
                "width": param_means[category][0].shape[0],
            }
            for side in [0, 1]
        ]
        for category in ["names", "countries"]
    ]
    param_list = [a for b in param_list for a in b]  # flatten
    return (
        parallel_inject(
            model=model,
            vocab=vocab,
            param_list=param_list,
            num_samples=num_samples,
            batch_size=50,
            num_entities=num_entities,
            context=vocab.default_context(num_entities=num_entities),
            base_selector_extractor=base_selector_extractor,
            promotions=[("countries", 1)],
            prompt_id_start=prompt_id_start,
            demotions=[],
            template_type=template_type,
            additional_template_context=additional_template_context,
            include_null=include_null,
        )
        .detach()
        .cpu()
    )  # names, batch, attr


def orthogonalize(tensor1, tensor2):
    """
    tensor1: [width, layer, dim]
    """
    vec1 = tensor1.flatten() / tensor1.norm()
    vec2 = tensor2.flatten()
    proj = torch.dot(vec1, vec2)
    return (vec2 - proj * vec1).reshape_as(tensor2)


def get_coeffs(base_means, tensor):
    vec = tensor.flatten()
    return [torch.dot(vec, mean.flatten()) / (mean.norm() ** 2) for mean in base_means]


def stack_tensors(names, countries):
    return torch.concatenate([names, countries])


def unstack_tensors(ten, entity_width):
    return ten[:entity_width], ten[entity_width:]


def sweep_grid(
    model,
    vocab,
    base_means,
    param_means,
    range0,
    range1,
    res0,
    res1,
    coeff_0,
    prompt_id_start,
    cross_countries,
    num_samples,
):
    xvals = np.linspace(range0[0], range0[1], res0)
    yvals = np.linspace(range1[0], range1[1], res1)
    return repeat(
        res0,
        lambda ix: repeat(
            res1,
            lambda iy: test_subspace(
                model=model,
                vocab=vocab,
                num_entities=2,
                base_means=base_means,
                param_means=param_means,
                coeffs=[coeff_0, [xvals[ix], yvals[iy]]],
                include_null=True,
                base_selector_extractor=default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
                cross_countries=cross_countries,
            ),
        ),
    )


def test_category_mean_interventions(
    *,
    model,
    vocab,
    num_entities,
    category,
    means,
    include_null,
    base_selector_extractor,
    template_type,
    num_samples,
    additional_template_context,
    prompt_id_start=0,
):
    all_swaps = rep.get_pairwise_switches(num_entities)

    def run_swap(swap_idx):
        swap = all_swaps[swap_idx]

        def obtain_intervention(means_list, base_idx, goal_idx):
            if base_idx == 0:
                return means_list[goal_idx - 1]
            elif goal_idx == 0:
                return -means_list[base_idx - 1]
            else:
                return means_list[goal_idx - 1] - means_list[base_idx - 1]

        param_list = [
            [
                {
                    "category": category,
                    "params": rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: obtain_intervention(
                                means[category], base_idx, goal_idx
                            )[:, layer, :]
                            for layer in range(model.cfg.n_layers)
                        },
                    ),
                    "side": base_idx,
                    "width": means[category][0].shape[0],
                }
                for base_idx, goal_idx in [swap, (swap[1], swap[0])]
            ]
        ]
        param_list = [a for b in param_list for a in b]  # flatten
        return parallel_inject(
            model=model,
            vocab=vocab,
            param_list=param_list,
            num_samples=num_samples,
            batch_size=50,
            num_entities=num_entities,
            context=vocab.default_context(num_entities=num_entities),
            base_selector_extractor=base_selector_extractor,
            promotions=[("countries", 1)] if category == "countries" else [],
            demotions=[],
            template_type=template_type,
            additional_template_context=additional_template_context,
            include_null=include_null,
            prompt_id_start=prompt_id_start,
        )

    return repeat(
        len(all_swaps), lambda swap_idx: run_swap(swap_idx)
    )  # swap_idx, names, batch, attr


def test_category_mean_interventions_cyclic(
    *,
    model,
    vocab,
    num_entities,
    category,
    base_means,
    param_means,
    include_null,
    base_selector_extractor,
    template_type,
    num_samples,
    additional_template_context,
    prompt_id_start=0,
):
    all_perms = rep.get_cyclic_perm(num_entities)

    def run_perm(perm_idx):
        perm = all_perms[perm_idx]  # list of new positions

        def obtain_intervention(base_means_list, param_means_list, base_idx, goal_idx):
            if base_idx == 0 and goal_idx == 0:
                return torch.zeros_like(base_means_list[0])
            elif base_idx == 0:
                return param_means_list[goal_idx - 1]
            elif goal_idx == 0:
                return -base_means_list[base_idx - 1]
            else:
                return param_means_list[goal_idx - 1] - base_means_list[base_idx - 1]

        param_list = [
            [
                {
                    "category": category,
                    "params": rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: obtain_intervention(
                                base_means[category],
                                param_means[category],
                                base_idx,
                                goal_idx,
                            )[:, layer, :]
                            for layer in range(model.cfg.n_layers)
                        },
                    ),
                    "side": base_idx,
                    "width": base_means[category][0].shape[0],
                }
                for base_idx, goal_idx in enumerate(perm)
            ]
        ]
        param_list = [a for b in param_list for a in b]  # flatten
        return parallel_inject(
            model=model,
            vocab=vocab,
            param_list=param_list,
            num_samples=num_samples,
            batch_size=50,
            num_entities=num_entities,
            context=vocab.default_context(num_entities=num_entities),
            base_selector_extractor=base_selector_extractor,
            promotions=[("countries", 1)] if category == "countries" else [],
            demotions=[],
            template_type=template_type,
            additional_template_context=additional_template_context,
            include_null=include_null,
            prompt_id_start=prompt_id_start,
        )

    return repeat(
        len(all_perms), lambda perm_idx: run_perm(perm_idx)
    )  # perm_idx, names, batch, attr


def test_cross_mean_interventions_cyclic_subspace(
    *,
    model,
    vocab,
    num_entities,
    base_means,
    param_means,
    include_null,
    base_selector_extractor,
    template_type,
    num_samples,
    additional_template_context,
    prompt_id_start=0,
):
    all_perms = rep.get_cyclic_perm(num_entities)

    def run_perm(perm_idx):
        perm = all_perms[perm_idx]  # list of new positions

        def obtain_intervention(base_means_list, param_means_list, base_idx, goal_idx):
            return param_means_list[goal_idx] - base_means_list[base_idx]

        shifted_category = "countries"
        param_list = [
            [
                {
                    "category": category,
                    "params": rep.UntiedInjectionParameters.from_dict(
                        model,
                        {
                            layer: obtain_intervention(
                                base_means[category],
                                param_means[category],
                                base_idx,
                                goal_idx if shifted_category == category else base_idx,
                            )[:, layer, :]
                            for layer in range(model.cfg.n_layers)
                        },
                    ),
                    "side": base_idx,
                    "width": base_means[category][0].shape[0],
                }
                for category in base_means.keys()
            ]
            for base_idx, goal_idx in enumerate(perm)
        ]
        param_list = [a for b in param_list for a in b]  # flatten
        return parallel_inject(
            model=model,
            vocab=vocab,
            param_list=param_list,
            num_samples=num_samples,
            batch_size=50,
            num_entities=num_entities,
            context=vocab.default_context(num_entities=num_entities),
            base_selector_extractor=base_selector_extractor,
            promotions=[("countries", 1)],
            demotions=[],
            template_type=template_type,
            additional_template_context=additional_template_context,
            include_null=include_null,
            prompt_id_start=prompt_id_start,
        )

    return repeat(
        len(all_perms), lambda perm_idx: run_perm(perm_idx)
    )  # perm_idx, names, batch, attr


def test_baseline(
    *,
    model,
    vocab,
    num_entities,
    include_null,
    template_type,
    additional_template_context,
    prompt_id_start,
    num_samples,
):
    return parallel_inject(
        model=model,
        vocab=vocab,
        param_list=[],
        num_samples=num_samples,
        batch_size=50,
        num_entities=num_entities,
        context=vocab.default_context(num_entities=num_entities),
        base_selector_extractor=None,
        promotions=[],
        demotions=[],
        template_type=template_type,
        additional_template_context=additional_template_context,
        include_null=include_null,
        prompt_id_start=prompt_id_start,
    )  # names, batch, attr


def mcq_parallel_exchange(
    *,
    model,
    vocab,
    num_entities,
    prompt_id_start,
    max_width,
    num_samples=100,
    batch_size=50,
    template_type="normal",
    additional_template_context,
    base_selector_extractor,
):
    """
    Swap A_a. B_b. <- B_a. A_b.
    """
    assert num_entities == 2
    # selector = rep.build_selector(vocab, num_entities)
    def evaluate(width):
        def data_selector_extractor(token_maps):
            sel = base_selector_extractor(token_maps)
            return {
                # By default, we have exclusive endpoints
                # In this case, the exclusive endpoint points to the '\n' belonging to the start of the next sentence
                # We want to include this, so we just don't correct for the exclusive endpoint
                "sentence": [
                    [positions - i for i in range(width)][::-1]
                    for positions in sel["sentence_ends"]
                ]
            }

        source_context = vocab.default_context(
            num_entities=2, labels=[1, 0], options=[0, 1]
        )
        contrast_reps, contrast_tokens, _ = ds.collect_data(
            model=model,
            selector=None,
            selector_extractor=data_selector_extractor,
            num_entities=num_entities,
            num_samples=num_samples,
            batch_size=batch_size,
            vocab=vocab,
            context=source_context,
            index_start=prompt_id_start,
            hook_name="resid_pre",
            template_type=template_type,
            template_context=dict(
                query_name=0, raw_query_name=None, **additional_template_context
            ),
        )
        if width > 0:
            for k, v in contrast_reps.items():
                for i in range(len(v)):
                    v[i] = einops.rearrange(
                        torch.stack(v[i]),
                        "width batch layer dim -> layer batch width dim",
                    )
        intervention_list = []
        exchange_ids = [(0, 0), (1, 1)]
        for source_id, target_id in exchange_ids:
            if width > 0:

                @lambda x: partial(x, target_id=target_id)
                def extractor(token_maps, target_id):
                    sel = data_selector_extractor(token_maps)
                    return sel["sentence"][target_id][
                        0
                    ]  # first token in sentence suffix

                param = rep.UntiedInjectionParameters.from_dict(
                    model,
                    {
                        layer: contrast_reps["sentence"][source_id][layer]
                        for layer in range(model.cfg.n_layers)
                    },
                )
                intervention_list.append(
                    (
                        param,
                        partial(param.batch_adapter, prompt_id_start=prompt_id_start),
                        partial(
                            rep.general_overwrite_hook_fn,
                            width=width,
                            extractor=extractor,
                        ),
                    )
                )
        compound_params, hook_fn_generator = compose_interventions(
            model=model, intervention_list=intervention_list
        )
        return repeat(
            num_entities,
            lambda query_name: rep.evaluate_params_refactored(
                model=model,
                parameters=compound_params,
                num_samples=num_samples,
                batch_size=batch_size,
                vocab=vocab,
                context=vocab.default_context(num_entities=num_entities),
                hook_fn_generator=hook_fn_generator,
                num_entities=num_entities,
                prompt_id_start=prompt_id_start,
                return_raw=True,
                template_type=template_type,
                template_context=dict(
                    query_name=query_name,
                    raw_query_name=None,
                    **additional_template_context,
                ),
            ),
        )

    return repeat(
        max_width + 1, lambda width: evaluate(width)
    )  # widths, name, batch, answer
