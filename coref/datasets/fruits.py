import pandas as pd
import os
from coref import COREF_ROOT
from collections import namedtuple
import itertools
from coref.datasets.common import BaseVocab
from coref.datasets.common import (
    Name,
    Attr,
    TrackFormatter,
    Substring,
    tokenize_prompt,
    recursive_add_offset,
    names,
)
from coref.utils import prepend

# df = pd.read_csv(os.path.join(COREF_ROOT, "coref/datasets/raw/colors.txt"), index_col=False, skiprows=[0], sep='\t', names=['color', 'hex'])
# colors = df.color.to_list()

df = pd.read_csv(os.path.join(COREF_ROOT, "coref/datasets/raw/fruits.csv"))
fruit_pairs = [(a, a) for a in df["fruits"].to_list()]


class Vocab(BaseVocab):
    type = "FRUITS"

    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        self.filtered_names = self.filter_names(names)
        self.filtered_country_capital_pairs = self.filter_countries(fruit_pairs)
        self.simple_train_test_split(split)


Statement = namedtuple("Statement", ["name", "attr", "type"])


class TEMPLATES:
    @classmethod
    def lookup(cls, template_type):
        if template_type == "normal":
            return cls.DEFAULT
        elif template_type == "noquestion":
            return cls.NOQUESTION
        raise Exception(f"unknown template {template_type}")

    class DEFAULT:
        @classmethod
        def lookup(cls, context_type):
            if context_type == "normal":
                return cls.context_template
            raise Exception(f"unknown context template {context_type}")

        @classmethod
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name):
                names_perm, attrs_perm = vocab.get_shuffled_labels(prompt_id)
                return (
                    cls.template,
                    dict(
                        qn_subject=vocab.filtered_names[names_perm[query_name]]
                        if raw_query_name is None
                        else raw_query_name,
                    ),
                )

            return ret

        @classmethod
        def extract_template_indices(cls, full_output_indices):
            return {
                "qn_subject": full_output_indices["qn_subject"][0],
                "ans_subject": full_output_indices["qn_subject"][-1],
            }

        @classmethod
        def instantiate(cls, vocab, statement, prompt_id, template_context):
            names_perm, attrs_perm = vocab.get_shuffled_labels(prompt_id)
            if statement.type == "normal" or statement.type == "ref":
                context_template = cls.lookup(statement.type)
                name = vocab.filtered_names[names_perm[statement.name]]
                country, capital = vocab.filtered_country_capital_pairs[
                    attrs_perm[statement.attr]
                ]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    subject=name,
                    country=country,
                )
                return (
                    cur_ctx,
                    {
                        "subject": ctx_idx_map["subject"][0],
                        "country": ctx_idx_map["country"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                raise Exception(f"unknown statement type {statement.type}")

        template = """Answer the question based on the context below. Keep the answer short.

Context:{context}

Question: Which fruit does {qn_subject} like?

Answer: {qn_subject} likes the"""
        context_template = """ {subject} likes eating the {country}."""

        def format_parallel(things):
            return ", ".join(things[:-1]) + " and " + things[-1]
