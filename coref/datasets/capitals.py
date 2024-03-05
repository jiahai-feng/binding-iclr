from coref.datasets.common import (
    names,
    country_capital_pairs,
    TrackFormatter,
    Substring,
    tokenize_prompt,
)
import os
import numpy as np
import itertools
from coref.utils import prepend
from coref.datasets.common import BaseVocab, Statement


class Vocab(BaseVocab):
    type = "CAPITALS"

    def __init__(self, tokenizer_type, split=None, is_parallel=False):
        super().__init__(tokenizer_type)

        self.filtered_names = self.filter_names(names)
        self.filtered_country_capital_pairs = self.filter_countries(
            country_capital_pairs
        )
        self.simple_train_test_split(split)
        self.is_parallel = is_parallel

    def default_context(self, num_entities, entities=None, attributes=None):
        if entities is None:
            entities = range(num_entities)
        if attributes is None:
            attributes = range(num_entities)
        if not self.is_parallel:
            return [Statement(e, s, "normal") for e, s in zip(entities, attributes)]
        else:
            return [Statement(list(entities), list(attributes), "parallel")]


class MAGIC_NUMBERS:
    CONTEXT_START = 18
    CONTEXT_END = 35
    ANSWER_START = 51
    FIRST_ATTR = 25
    SECOND_ATTR = 34


class MAGIC_NUMBERS_1:
    CONTEXT_START = 18
    CONTEXT_END = 26
    ANSWER_START = 42
    FIRST_ATTR = 25


class TEMPLATES:
    @classmethod
    def lookup(cls, s):
        if s == "normal":
            return cls.DEFAULT
        elif s == "work":
            return cls.WORK

    class DEFAULT:
        @classmethod
        def lookup(cls, s):
            if s == "normal":
                return cls.context_template
            elif s == "alt":
                return cls.alt_context_template
            raise Exception(f"unknown context template {s}")

        template = """Answer the question based on the context below. Keep the answer short.

Context:{context}

Question: Which city does {qn_subject} live in?

Answer: {qn_subject} lives in the city of"""

        context_template = """ {subject} lives in the capital city of {country}."""

        alt_context_template = (
            """ {subject} currently resides in the capital city of {country}."""
        )

    class WORK:
        @classmethod
        def lookup(cls, s):
            if s == "normal":
                return cls.context_template
            raise Exception(f"unknown context template {s}")

        template = """Answer the question based on the context below. Keep the answer short.

Context:{context}

Question: Which city does {qn_subject} work in?

Answer: {qn_subject} works in the city of"""

        context_template = """ {subject} works in the capital city of {country}."""


def generate_prompts(
    *,
    vocab,
    context,
    query_name,
    prompt_id=None,
    template_type="normal",
    context_type="normal",
    **kwargs,
):
    """
    Args:
        vocab: Vocab
        context: List[Tuple[Name, Attr] or Statement]
        template_type: "normal" or "work"
        context_type: "normal" or "alt"
    """
    context_template = TEMPLATES.lookup(template_type).lookup(context_type)
    template = TEMPLATES.lookup(template_type).template
    names_perm, attrs_perm = vocab.get_shuffled_labels(prompt_id)
    context_text = []
    context_indices = []
    for name_id, attr_id in context:
        name = vocab.filtered_names[names_perm[name_id]]
        country, capital = vocab.filtered_country_capital_pairs[attrs_perm[attr_id]]
        cur_ctx, ctx_idx_map = TrackFormatter().format(
            context_template,
            subject=name,
            country=country,
        )
        context_text.append(cur_ctx)
        context_indices.append(
            {
                "subject": ctx_idx_map["subject"][0],
                "country": ctx_idx_map["country"][0],
                "sentence": Substring(0, len(cur_ctx)),
            }
        )
    joined_context_text = "".join(context_text)
    full_output, full_output_indices = TrackFormatter().format(
        template,
        context=joined_context_text,
        qn_subject=vocab.filtered_names[names_perm[query_name]],
    )
    context_start_pos = full_output_indices["context"][0].start
    context_acc = itertools.accumulate(
        [context_start_pos] + [len(x) for x in context_text]
    )
    full_output_indices = {
        "context_section": full_output_indices["context"][0],
        "context": [
            {key: v + offset for key, v in ctx_idx_map.items()}
            for offset, ctx_idx_map in zip(context_acc, context_indices)
        ],
        "qn_subject": full_output_indices["qn_subject"][0],
        "ans_subject": full_output_indices["qn_subject"][-1],
    }
    return dict(prompt=full_output, indices=full_output_indices)


generate_tokenized_prompts = prepend(generate_prompts)(tokenize_prompt)
