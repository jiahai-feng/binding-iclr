import itertools
import coref.datasets.shapes as shapes
import coref.datasets.capitals as capitals
import coref.datasets.fruits as fruits
import coref.datasets.compound_capitals as compound_capitals
import coref.datasets.mcq_agnews as mcq_agnews
import coref.datasets.biosbias as biosbias
import coref.datasets.icl_sst as icl_sst
import coref.datasets.mcq_sst as mcq_sst
import coref.datasets.onehop as onehop
from coref.utils import prepend
from coref.datasets.common import (
    tokenize_prompt,
    Substring,
    recursive_add_offset,
    TrackFormatter,
)


VOCAB_MAP = {
    "CAPITALS": capitals.Vocab,
    "SHAPES": shapes.Vocab,
    "FRUITS": fruits.Vocab,
    "MCQ_AGNEWS": mcq_agnews.Vocab,
    "BIOSBIAS": biosbias.Vocab,
    "ICL_SST": icl_sst.Vocab,
    "MCQ_SST": mcq_sst.Vocab,
    'ONEHOP': onehop.Vocab
}
TEMPLATE_MAP = {
    "CAPITALS": compound_capitals.TEMPLATES,
    "SHAPES": shapes.TEMPLATES,
    "FRUITS": fruits.TEMPLATES,
    "MCQ_AGNEWS": mcq_agnews.TEMPLATES,
    "BIOSBIAS": biosbias.TEMPLATES,
    "ICL_SST": icl_sst.TEMPLATES,
    "MCQ_SST": mcq_sst.TEMPLATES,
    'ONEHOP': onehop.TEMPLATES
}


def load_vocab(vocab_tag, model_tag, **vocab_kwargs):
    if "llama" in model_tag or "tulu" in model_tag:
        tokenizer_type = "llama"
    else:
        tokenizer_type = "pythia"
    if vocab_tag in VOCAB_MAP:
        return VOCAB_MAP[vocab_tag](tokenizer_type, **vocab_kwargs)
    elif vocab_tag == "PARALLEL":
        return VOCAB_MAP["CAPITALS"](tokenizer_type, is_parallel=True, **vocab_kwargs)
    else:
        raise Exception(f"Unknown vocab tag {vocab_tag}")


def generate_prompts(
    *,
    vocab,
    context,
    query_name=None,
    prompt_id=None,
    template_type="normal",
    template_context=None,
    raw_query_name=None,
    **kwargs,
):
    """
    Using setting raw_query_name overrides query_name
    Args:
        tokenizer
        context: List[Statement]
        query_name: Name
        raw_query_name: None | str
    """
    if prompt_id is None:
        prompt_id = vocab.global_rng.integers(int(1e18))
    if template_context is None:
        template_context = dict(query_name=query_name, raw_query_name=raw_query_name)

    if vocab.type in VOCAB_MAP:
        TEMPLATES = TEMPLATE_MAP[vocab.type]
    else:
        raise Exception(f"Unknown vocab type {vocab.type}")
    template = TEMPLATES.lookup(template_type)
    template_string, template_substitutions = template.generate_template(
        template_context=template_context,
        prompt_id=prompt_id,
        vocab=vocab,
        context=context,
    )
    context_text = []
    context_indices = []
    for statement in context:
        text, indices = template.instantiate(
            vocab=vocab,
            statement=statement,
            prompt_id=prompt_id,
            template_context=template_context,
        )
        context_text.append(text)
        context_indices.append(indices)
    joined_context_text = "".join(context_text)
    full_output, full_output_indices = TrackFormatter().format(
        template_string, context=joined_context_text, **template_substitutions
    )
    context_start_pos = full_output_indices["context"][0].start
    context_acc = itertools.accumulate(
        [context_start_pos] + [len(x) for x in context_text]
    )
    full_output_indices = {
        "prompt": Substring(0, len(full_output)),
        "context_section": full_output_indices["context"][0],
        "context": [
            recursive_add_offset(ctx_idx_map, offset)
            for offset, ctx_idx_map in zip(context_acc, context_indices)
        ],
        **template.extract_template_indices(full_output_indices),
    }
    return dict(prompt=full_output, indices=full_output_indices)


def generate_options(labels, options):
    assert len(labels) == len(options)
    option_strings = [f"\n{label}: option" for label, option in zip(labels, options)]
    return "".join(option_strings)


generate_tokenized_prompts = prepend(generate_prompts)(tokenize_prompt)
