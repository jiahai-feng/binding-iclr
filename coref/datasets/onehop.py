import pandas as pd
import numpy as np
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
    colors,
    names,
    country_capital_pairs,
    train_test_split
)
from coref.utils import prepend


class Vocab(BaseVocab):
    type = "ONEHOP"

    def __init__(self, tokenizer_type, split=None):
        super().__init__(tokenizer_type)

        
        self.filtered_names = self.filter_names(names)
        self.filtered_country_capital_pairs = self.filter_countries(
            country_capital_pairs
        )
        self.simple_train_test_split(split)


        self.filtered_colors = self.filter_names(colors)
        # manually do train/test split
        if split == 'train':
            self.filtered_colors, _ = train_test_split(self.filtered_colors, 0.5)

    def get_shuffled_labels(self, prompt_id=None):
        return self.generate_permutations(
            prompt_id,
            [len(self.filtered_names), len(self.filtered_country_capital_pairs), len(self.filtered_colors)],
        )

    def fetch_shuffled_name(self, name_id, prompt_id):
        names_perm, _, _ = self.get_shuffled_labels(prompt_id)
        return self.filtered_names[names_perm[name_id]]

    def fetch_shuffled_attr(self, attr_id, prompt_id):
        _, attrs_perm, colors_perm = self.get_shuffled_labels(prompt_id)
        return (self.filtered_country_capital_pairs[attrs_perm[attr_id]][0], self.filtered_colors[colors_perm[attr_id]])



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
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name, num_qn_context):
                names_perm, attrs_perm, colors_perm = vocab.get_shuffled_labels(prompt_id)
                rng = np.random.default_rng(prompt_id + 93843029)
                qn_context = ''.join(rng.permutation([
                    cls.qn_context_template.format(
                        city=vocab.filtered_country_capital_pairs[attrs_perm[i]][1],
                        color=vocab.filtered_colors[colors_perm[i]]
                    )
                    for i in range(num_qn_context)
                ]).tolist())
                return (
                    cls.template,
                    dict(
                        qn_context=qn_context,
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
        def lookup(cls, context_type):
            if context_type == "normal":
                return cls.context_template
            raise Exception(f"unknown context template {context_type}")

        @classmethod
        def instantiate(cls, vocab, statement, prompt_id, template_context):
            names_perm, attrs_perm, colors_perm = vocab.get_shuffled_labels(prompt_id)
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

Question:{qn_context} What color does {qn_subject} like?

Answer: {qn_subject} likes"""
        context_template = """ {subject} lives in the capital city of {country}."""
        qn_context_template = ''' The person living in {city} likes {color}.'''
        def format_parallel(things):
            return ", ".join(things[:-1]) + " and " + things[-1]
