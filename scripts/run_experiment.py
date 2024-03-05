import os
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial

import torch

import coref.models as models
import coref.datascience as ds
import coref.data_collection as dc
from coref.datasets.api import load_vocab


def run_factorizability(
    model, prompt_id_start, cache, num_samples=100, entity_width=2, country_width=1, onehop_only=False
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    additional_template_context = {}
    if onehop_only:
        vocab_list = ['ONEHOP']
        additional_template_context = {'num_qn_context': 4}
    for vocab_tag in vocab_list:
        vocab = load_vocab(vocab_tag, model.model_tag)
        cache(
            f"factorizability_{vocab_tag}_side_{0}_{entity_width}_{country_width}",
            lambda: dc.factorization_parallel_exchange_by_side(
                model=model,
                vocab=vocab,
                num_entities=2,
                side=0,
                prompt_id_start=prompt_id_start,
                widths={"countries": country_width, "names": entity_width},
                num_samples=num_samples,
                batch_size=50,
                template_type="normal",
                additional_template_context=additional_template_context,
                base_selector_extractor=dc.default_binding_selector_extractor,
            ),
        )
        cache(
            f"factorizability_{vocab_tag}_side_{1}_{entity_width}_{country_width}", # used to be 'factorizability_{vocab_tag}_side_{1}_entities_{entity_width}'
            lambda: dc.factorization_parallel_exchange_by_side(
                model=model,
                vocab=vocab,
                num_entities=2,
                side=1,
                prompt_id_start=prompt_id_start,
                widths={"countries": country_width, "names": entity_width},
                num_samples=num_samples,
                batch_size=50,
                template_type="normal",
                additional_template_context=additional_template_context,
                base_selector_extractor=dc.default_binding_selector_extractor,
            ),
        )
        # cache(f'factorizability_{vocab_tag}_countries_{country_width}', lambda: dc.factorization_parallel_exchange(
        #     model=model,
        #     vocab=vocab,
        #     num_entities=2,
        #     prompt_id_start=prompt_id_start,
        #     widths={'countries': country_width},
        #     num_samples=num_samples,
        #     batch_size=50,
        #     template_type='normal',
        #     additional_template_context={},
        #     base_selector_extractor=dc.default_binding_selector_extractor
        # ))
        # cache(f'factorizability_{vocab_tag}_entities_{entity_width}', lambda: dc.factorization_parallel_exchange(
        #     model=model,
        #     vocab=vocab,
        #     num_entities=2,
        #     prompt_id_start=prompt_id_start,
        #     widths={'names': entity_width},
        #     num_samples=num_samples,
        #     batch_size=50,
        #     template_type='normal',
        #     additional_template_context={},
        #     base_selector_extractor=dc.default_binding_selector_extractor
        # ))


def run_position_independence(
    model, prompt_id_start, cache, num_samples=100, entity_width=2, country_width=1, normalize_logits=True, onehop_only=False
):
    vocab_list = [
        "CAPITALS",
    ]
    additional_template_context = {}
    if onehop_only:
        vocab_list = ['ONEHOP']
        additional_template_context = {'num_qn_context': 2}
    for vocab_tag in vocab_list:
        vocab = load_vocab(vocab_tag, model.model_tag)
        cache(
            f"position_{vocab_tag}_countries_{country_width}_normalize_{normalize_logits}",
            lambda: dc.position_intervention(
                model=model,
                vocab=vocab,
                category="countries",
                width=country_width,
                num_entities=2,
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
                template_type="normal",
                additional_template_context=additional_template_context,
                base_selector_extractor=dc.default_binding_selector_extractor,
                normalize_logits=normalize_logits
            ),
        )
        cache(
            f"position_{vocab_tag}_entities_{entity_width}_normalize_{normalize_logits}",
            lambda: dc.position_intervention(
                model=model,
                vocab=vocab,
                category="names",
                width=entity_width,
                num_entities=2,
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
                template_type="normal",
                additional_template_context=additional_template_context,
                base_selector_extractor=dc.default_binding_selector_extractor,
                normalize_logits=normalize_logits
            ),
        )

def run_local_position_independence(
    model, prompt_id_start, cache, num_samples=100, entity_width=2, country_width=1, normalize_logits=True, onehop_only=False
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    additional_template_context = {}
    if onehop_only:
        vocab_list = ['ONEHOP']
        additional_template_context = {'num_qn_context': 2}
    for is_tail in [True, False]:
        for vocab_tag in vocab_list:
            vocab = load_vocab(vocab_tag, model.model_tag)
            cache(
                f"local_position_tail_{is_tail}_{vocab_tag}_countries_{country_width}_normalize_{normalize_logits}",
                lambda: dc.position_intervention(
                    model=model,
                    vocab=vocab,
                    category="countries",
                    width=country_width,
                    num_entities=2,
                    prompt_id_start=prompt_id_start,
                    num_samples=num_samples,
                    template_type="normal",
                    additional_template_context=additional_template_context,
                    base_selector_extractor=dc.default_binding_selector_extractor,
                    normalize_logits=normalize_logits,
                    fixed_offset_start=-2,
                    fixed_offset_end=3,
                    is_tail=is_tail
                ),
            )
            cache(
                f"local_position_tail_{is_tail}_{vocab_tag}_entities_{entity_width}_normalize_{normalize_logits}",
                lambda: dc.position_intervention(
                    model=model,
                    vocab=vocab,
                    category="names",
                    width=entity_width,
                    num_entities=2,
                    prompt_id_start=prompt_id_start,
                    num_samples=num_samples,
                    template_type="normal",
                    additional_template_context=additional_template_context,
                    base_selector_extractor=dc.default_binding_selector_extractor,
                    normalize_logits=normalize_logits,
                    fixed_offset_start=-2,
                    fixed_offset_end=3,
                    is_tail=is_tail
                ),
            )

def random_vector_like(tens):
    norms = torch.linalg.vector_norm(tens, dim=-1, keepdim=True)
    noise = torch.normal(torch.zeros_like(tens), 1)
    noise /= torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    return noise * norms
def run_mean_interventions(
    *,
    model,
    prompt_id_start,
    cache,
    num_mean_samples=500,
    num_samples=100,
    entity_width=2,
    country_width=1,
    is_baseline=False,
    onehop_only=False,
):
    vocab_list = ["CAPITALS"]
    if onehop_only:
        vocab_list = ['ONEHOP']
    num_entities = 2
    for vocab_tag in vocab_list:
        if vocab_tag == 'ONEHOP':
            default_context = {'num_qn_context': 3}
            print('onehop mean int')
        else:
            default_context = {}
        test_vocab = load_vocab(vocab_tag, model.model_tag, split="test")
        train_vocab = load_vocab(vocab_tag, model.model_tag, split="train")
        vocab_means = cache(
            f"{vocab_tag}_{num_entities}_means",
            lambda: ds.extract_k_means(
                model=model,
                vocab=train_vocab,
                base_selector_extractor=dc.default_binding_selector_extractor,
                widths={"names": entity_width, "countries": country_width},
                num_entities=num_entities,
                prompt_id_start=prompt_id_start,
                num_samples=num_mean_samples,
                additional_template_context=default_context
            ),
        )
        if is_baseline:
            vocab_means = {
                category: [random_vector_like(mean) for mean in means]
                for category, means in vocab_means.items()
            }
        cache(
            f"mean_interventions_{vocab_tag}_{entity_width}_{country_width}" + ("_baseline" if is_baseline else ""),
            lambda: dc.test_mean_interventions(
                model=model,
                vocab=test_vocab,
                num_entities=num_entities,
                means=vocab_means,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                include_null=True,
                num_samples=num_samples,
                additional_template_context=default_context
            ),
        )


def run_subspace(
    *,
    model,
    prompt_id_start,
    cache,
    num_subspace_samples=50,
    num_mean_samples=500,
    entity_width=2,
    country_width=1,
    cross_countries
):
    vocab_list = ["CAPITALS"]
    num_entities = 3
    res = 10
    for vocab_tag in vocab_list:
        test_vocab = load_vocab(vocab_tag, model.model_tag, split="test")
        train_vocab = load_vocab(vocab_tag, model.model_tag, split="train")
        vocab_means = cache(
            f"{vocab_tag}_{num_entities}_means",
            lambda: ds.extract_k_means(
                model=model,
                vocab=train_vocab,
                base_selector_extractor=dc.default_binding_selector_extractor,
                widths={"names": entity_width, "countries": country_width},
                num_entities=num_entities,
                num_samples=num_mean_samples,
                prompt_id_start=prompt_id_start,
            ),
        )

        stack0, stack1 = [
            dc.stack_tensors(vocab_means["names"][i], vocab_means["countries"][i])
            for i in range(2)
        ]

        ortho = dc.orthogonalize(stack0, stack1)

        coefficients = cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_coeff",
            lambda: dc.get_coeffs([stack0, ortho], stack1),
        )
        print(f"Coefficients from orthogonalizing subspace: {coefficients}")

        ortho_names, ortho_countries = dc.unstack_tensors(ortho, entity_width)

        param_means = {
            "names": [vocab_means["names"][0], ortho_names],
            "countries": [vocab_means["countries"][0], ortho_countries],
        }
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0_0",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0, 0],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_1_0",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[1, 0],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0_1",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0, 1],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_1_1",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[1, 1],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0.5_0.5",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0.5, 0.5],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0.2_0.1",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0.2, 0.1],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0.5_0.7",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0.5, 0.7],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0.7_0.1",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0.7, 0.1],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )
        cache(
            f"subspace_{vocab_tag}_{cross_countries}_{entity_width}_{country_width}_0_0.5",
            lambda: dc.sweep_grid(
                model=model,
                vocab=test_vocab,
                base_means=vocab_means,
                param_means=param_means,
                prompt_id_start=prompt_id_start,
                range0=[-0.5, 1.5],
                range1=[-0.5, 1.5],
                res0=res,
                res1=res,
                coeff_0=[0., 0.5],
                cross_countries=cross_countries,
                num_samples=num_subspace_samples,
            ),
        )

def run_baseline(
    *,
    model,
    prompt_id_start,
    cache,
    capitals_only,
    num_samples,
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    if capitals_only:
        vocab_list = ["CAPITALS"]
    num_entities = 3
    for vocab_tag in vocab_list:
        test_vocab = load_vocab(vocab_tag, model.model_tag, split="test")
    
        cache(
            f"baseline_{vocab_tag}",
            lambda: dc.test_baseline(
                model=model,
                vocab=test_vocab,
                num_entities=num_entities,
                include_null=True,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )

def run_category_means_interventions(
    model,
    prompt_id_start,
    cache,
    capitals_only,
    num_mean_samples=500,
    num_samples=100,
    entity_width=2,
    country_width=1,
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    if capitals_only:
        vocab_list = ["CAPITALS"]
    num_entities = 3
    for vocab_tag in vocab_list:
        test_vocab = load_vocab(vocab_tag, model.model_tag, split="test")
        train_vocab = load_vocab(vocab_tag, model.model_tag, split="train")
        vocab_means = cache(
            f"{vocab_tag}_{num_entities}_means",
            lambda: ds.extract_k_means(
                model=model,
                vocab=train_vocab,
                base_selector_extractor=dc.default_binding_selector_extractor,
                widths={"names": entity_width, "countries": country_width},
                num_entities=num_entities,
                num_samples=num_mean_samples,
                prompt_id_start=prompt_id_start,
            ),
        )

        cache(
            f"mean_countries_interventions_{vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions(
                model=model,
                vocab=test_vocab,
                num_entities=3,
                category="countries",
                means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )
        cache(
            f"mean_entities_interventions_{vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions(
                model=model,
                vocab=test_vocab,
                num_entities=3,
                category="names",
                means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )


def run_cross_means_intervention(
    model,
    prompt_id_start,
    cache,
    num_mean_samples=500,
    num_samples=100,
    entity_width=2,
    country_width=1,
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    target_vocab_tag = "CAPITALS"
    target_vocab = load_vocab(target_vocab_tag, model.model_tag)
    num_entities = 3
    for source_vocab_tag in vocab_list:
        source_vocab = load_vocab(source_vocab_tag, model.model_tag)
        vocab_means = cache(
            f"{source_vocab_tag}_{num_entities}_means_spread",
            lambda: ds.extract_k_means(
                model=model,
                vocab=source_vocab,
                base_selector_extractor=dc.default_binding_selector_extractor,
                widths={"names": entity_width, "countries": country_width},
                num_entities=num_entities,
                num_samples=num_mean_samples,
                prompt_id_start=prompt_id_start,
                spread=True,
            ),
        )

        cache(
            f"mean_countries_cross_{source_vocab_tag}_{target_vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions(
                model=model,
                vocab=target_vocab,
                num_entities=num_entities,
                category="countries",
                means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )
        cache(
            f"mean_entities_cross_{source_vocab_tag}_{target_vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions(
                model=model,
                vocab=target_vocab,
                num_entities=num_entities,
                category="names",
                means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )

def run_category_means_interventions_cyclic(
    model,
    prompt_id_start,
    cache,
    capitals_only,
    onehop_only=False,
    num_mean_samples=500,
    num_samples=100,
    entity_width=2,
    country_width=1,
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    if capitals_only:
        vocab_list = ["CAPITALS"]
    if onehop_only:
        vocab_list = ['ONEHOP']
    num_entities = 3
    for vocab_tag in vocab_list:
        if vocab_tag == 'ONEHOP':
            default_context = {'num_qn_context': 3}
            print('onehop cyclic mean int')
        else:
            default_context = {}
        test_vocab = load_vocab(vocab_tag, model.model_tag, split="test")
        train_vocab = load_vocab(vocab_tag, model.model_tag, split="train")
        vocab_means = cache(
            f"{vocab_tag}_{num_entities}_means",
            lambda: ds.extract_k_means(
                model=model,
                vocab=train_vocab,
                base_selector_extractor=dc.default_binding_selector_extractor,
                widths={"names": entity_width, "countries": country_width},
                num_entities=num_entities,
                num_samples=num_mean_samples,
                prompt_id_start=prompt_id_start,
                additional_template_context=default_context
            ),
        )

        cache(
            f"mean_countries_interventions_cyclic_{vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions_cyclic(
                model=model,
                vocab=test_vocab,
                num_entities=3,
                category="countries",
                base_means=vocab_means,
                param_means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context=default_context,
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )
        cache(
            f"mean_entities_interventions_cyclic_{vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions_cyclic(
                model=model,
                vocab=test_vocab,
                num_entities=3,
                category="names",
                base_means=vocab_means,
                param_means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context=default_context,
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )


def run_cross_means_intervention_cyclic(
    model,
    prompt_id_start,
    cache,
    num_mean_samples=500,
    num_samples=100,
    entity_width=2,
    country_width=1,
):
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS"]
    target_vocab_tag = "CAPITALS"
    target_vocab = load_vocab(target_vocab_tag, model.model_tag)
    num_entities = 3
    for source_vocab_tag in vocab_list:
        source_vocab = load_vocab(source_vocab_tag, model.model_tag)
        vocab_means = cache(
            f"{source_vocab_tag}_{num_entities}_means_spread",
            lambda: ds.extract_k_means(
                model=model,
                vocab=source_vocab,
                base_selector_extractor=dc.default_binding_selector_extractor,
                widths={"names": entity_width, "countries": country_width},
                num_entities=num_entities,
                num_samples=num_mean_samples,
                prompt_id_start=prompt_id_start,
                spread=True,
            ),
        )

        cache(
            f"mean_countries_cross_cyclic_{source_vocab_tag}_{target_vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions_cyclic(
                model=model,
                vocab=target_vocab,
                num_entities=num_entities,
                category="countries",
                base_means=vocab_means,
                param_means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )
        cache(
            f"mean_entities_cross_cyclic_{source_vocab_tag}_{target_vocab_tag}_{entity_width}_{country_width}",
            lambda: dc.test_category_mean_interventions_cyclic(
                model=model,
                vocab=target_vocab,
                num_entities=num_entities,
                category="names",
                base_means=vocab_means,
                param_means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )


def run_cross_mean_interventions_cyclic_subspace(
    model,
    prompt_id_start,
    cache,
    num_mean_samples=500,
    num_samples=100,
    entity_width=2,
    country_width=1,
):
    spread=False
    vocab_list = ["CAPITALS", "PARALLEL", "SHAPES", "FRUITS", "BIOSBIAS", 'ZEROS', 'RANDOM']
    target_vocab_tag = "CAPITALS"
    target_vocab = load_vocab(target_vocab_tag, model.model_tag)
    num_entities = 3
    target_means = cache(
        f"{target_vocab_tag}_{num_entities}_means",
        lambda: ds.extract_k_means(
            model=model,
            vocab=target_vocab,
            base_selector_extractor=dc.default_binding_selector_extractor,
            widths={"names": entity_width, "countries": country_width},
            num_entities=num_entities,
            num_samples=num_mean_samples,
            prompt_id_start=prompt_id_start,
            spread=False,
        ),
    )
    target_means = {
        category: [torch.zeros_like(means[0])] + means
        for category, means in target_means.items()
    }
    for source_vocab_tag in vocab_list:
        if source_vocab_tag == 'ZEROS':
            vocab_means = {
                category: [torch.zeros_like(mean) for mean in means]
                for category, means in target_means.items()
            }
        elif source_vocab_tag == 'RANDOM':
            vocab_means = {
                category: [random_vector_like(mean) for mean in means]
                for category, means in target_means.items()
            }
        else:
            source_vocab = load_vocab(source_vocab_tag, model.model_tag)
            vocab_means = cache(
                f"{source_vocab_tag}_{num_entities}_means_spread_{spread}",
                lambda: ds.extract_k_means(
                    model=model,
                    vocab=source_vocab,
                    base_selector_extractor=dc.default_binding_selector_extractor,
                    widths={"names": entity_width, "countries": country_width},
                    num_entities=num_entities + 1, # we extract 3 diff vectors
                    num_samples=num_mean_samples,
                    prompt_id_start=prompt_id_start,
                    spread=spread,
                ),
            )
        print('diff')
        cache(
            f"mean_cross_cyclic_subspace_{source_vocab_tag}_{target_vocab_tag}_{entity_width}_{country_width}_{spread}",
            lambda: dc.test_cross_mean_interventions_cyclic_subspace(
                model=model,
                vocab=target_vocab,
                num_entities=num_entities,
                base_means=target_means,
                param_means=vocab_means,
                include_null=True,
                base_selector_extractor=dc.default_binding_selector_extractor,
                template_type="normal",
                additional_template_context={},
                prompt_id_start=prompt_id_start,
                num_samples=num_samples,
            ),
        )

def run_mcq(
    *,
    model,
    prompt_id_start,
    cache,
    num_samples=100,
    max_width
):
    vocab_tag = 'MCQ_SST'
    vocab = load_vocab(vocab_tag, model.model_tag)
    cache(f'mcq_{vocab_tag}_width_{max_width}', lambda: dc.mcq_parallel_exchange(
        model=model,
        vocab=vocab,
        num_entities=vocab.dataset.num_options(),
        prompt_id_start=prompt_id_start,
        max_width=max_width,
        num_samples=num_samples,
        batch_size=50,
        template_type="reversed",
        additional_template_context={},
        base_selector_extractor=dc.statements_mcq_selector_extractor,
    ))
    

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help=f'choices: {", ".join(models.PYTHIA_MODELS + models.LLAMA_MODELS + models.TULU_MODELS)}',
        required=True,
    )
    parser.add_argument("--num_devices", type=int, default=1)
    # things
    parser.add_argument("--factorizability", action="store_true")
    parser.add_argument("--position", action="store_true")
    parser.add_argument("--local_position", action="store_true")
    parser.add_argument("--means_intervention", action="store_true")
    parser.add_argument("--means_intervention_baseline", action="store_true")
    parser.add_argument("--subspace", action="store_true")
    parser.add_argument("--force_time", nargs="?", type=str)
    parser.add_argument("--all_tasks_means", action="store_true")
    parser.add_argument("--cross_tasks_means", action="store_true")
    parser.add_argument("--capitals_means", action="store_true")
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--capitals_baseline', action='store_true')
    parser.add_argument('--mcq', action='store_true')
    parser.add_argument("--cross_tasks_subspace", action="store_true")
    # options
    parser.add_argument("--prompt_id_start", type=int, default=0)
    parser.add_argument("--cache_path", default=os.path.join(COREF_ROOT, "cache"))
    parser.add_argument("--force_compute", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--entity_width", type=int, default=2)
    parser.add_argument("--country_width", type=int, default=1)
    parser.add_argument('--no_normalize_logits', action='store_true')
    parser.add_argument('--cross_countries', action='store_true')
    parser.add_argument('--cyclic', action='store_true')
    parser.add_argument('--onehop', action='store_true')

    args = parser.parse_args()

    import time

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if args.force_time is not None:
        timestr = args.force_time
    print(f"Using cache path {args.cache_path}")
    print(f"Using timestamp {timestr}")
    print(f'Force_compute: {args.force_compute}')
    eval_cache = partial(
        cache,
        prefixes=[args.cache_path, "paper_data", args.model, timestr]
        + (["dry_run"] if args.dry_run else []),
        force_compute=args.force_compute,
    )
    model = models.fetch_model(args.model, num_devices=args.num_devices)

    if args.dry_run:
        num_samples = 2
        num_mean_samples = 2
        num_subspace_samples = 2
    else:
        num_samples = 100
        num_mean_samples = 500
        num_subspace_samples = 50

    if args.factorizability:
        print("Running factorizability")
        run_factorizability(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
        )
    if args.position:
        print("Running position")
        run_position_independence(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            normalize_logits=not args.no_normalize_logits
        )
    if args.local_position:
        print("Running local position")
        run_local_position_independence(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            normalize_logits=not args.no_normalize_logits
        )
    if args.means_intervention:
        print("Running means intervention")
        run_mean_interventions(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            num_mean_samples=num_mean_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
        )
    if args.means_intervention_baseline:
        print("Running means intervention baseline")
        run_mean_interventions(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            num_mean_samples=num_mean_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            is_baseline=True
        )
    if args.subspace:
        print("Running subspace")
        run_subspace(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_subspace_samples=num_subspace_samples,
            num_mean_samples=num_mean_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            cross_countries=args.cross_countries
        )
    if args.all_tasks_means:
        print("Running all tasks means")
        if args.cyclic:
            run_category_means_interventions_cyclic(
                model=model,
                prompt_id_start=args.prompt_id_start,
                cache=eval_cache,
                capitals_only=False,
                num_samples=num_samples,
                num_mean_samples=num_mean_samples,
                entity_width=args.entity_width,
                country_width=args.country_width,
            )
        else:
            run_category_means_interventions(
                model=model,
                prompt_id_start=args.prompt_id_start,
                cache=eval_cache,
                capitals_only=False,
                num_samples=num_samples,
                num_mean_samples=num_mean_samples,
                entity_width=args.entity_width,
                country_width=args.country_width,
            )
    if args.capitals_means:
        print("Running capitals means")
        if args.cyclic:
            run_category_means_interventions_cyclic(
                model=model,
                prompt_id_start=args.prompt_id_start,
                cache=eval_cache,
                capitals_only=True,
                num_samples=num_samples,
                num_mean_samples=num_mean_samples,
                entity_width=args.entity_width,
                country_width=args.country_width,
            )
        else:
            run_category_means_interventions(
                model=model,
                prompt_id_start=args.prompt_id_start,
                cache=eval_cache,
                capitals_only=True,
                num_samples=num_samples,
                num_mean_samples=num_mean_samples,
                entity_width=args.entity_width,
                country_width=args.country_width,
            )
    if args.cross_tasks_means:
        print("Running cross task means")
        if args.cyclic:
            run_cross_means_intervention_cyclic(
                model=model,
                prompt_id_start=args.prompt_id_start,
                cache=eval_cache,
                num_samples=num_samples,
                num_mean_samples=num_mean_samples,
                entity_width=args.entity_width,
                country_width=args.country_width,
            )
        else:
            run_cross_means_intervention(
                model=model,
                prompt_id_start=args.prompt_id_start,
                cache=eval_cache,
                num_samples=num_samples,
                num_mean_samples=num_mean_samples,
                entity_width=args.entity_width,
                country_width=args.country_width,
            )
        
    if args.cross_tasks_subspace:
        print("Running cross task means")
        if not args.cyclic:
            print('Warning, args.cyclic set to false, but running cyclic cross subspace anyway')
        run_cross_mean_interventions_cyclic_subspace(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            num_mean_samples=num_mean_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
        )
    if args.baseline:
        run_baseline(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            capitals_only=False
        )
    if args.capitals_baseline:
        run_baseline(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            capitals_only=True
        )
    if args.mcq:
        run_mcq(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            max_width=5
        )
    if args.onehop:
        print("Running onehop")
        run_factorizability(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            onehop_only=True
        )
        run_local_position_independence(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            normalize_logits=not args.no_normalize_logits,
            onehop_only=True
        )
        run_mean_interventions(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            num_samples=num_samples,
            num_mean_samples=num_mean_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
            onehop_only=True
        )
        run_category_means_interventions_cyclic(
            model=model,
            prompt_id_start=args.prompt_id_start,
            cache=eval_cache,
            capitals_only=False,
            onehop_only=True,
            num_samples=num_samples,
            num_mean_samples=num_mean_samples,
            entity_width=args.entity_width,
            country_width=args.country_width,
        )
    print('done')
    


if __name__ == "__main__":
    main()
