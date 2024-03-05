import pandas as pd
from pathlib import Path
import os
from coref import COREF_ROOT
from collections import namedtuple
import itertools
from coref.datasets.common import BaseVocab
import coref.datasets.biasinbios_process as bib
from coref.datasets.common import (
    Name,
    Attr,
    TrackFormatter,
    Substring,
    train_test_split,
)

processed_json_dataset_file = Path(COREF_ROOT) / "coref/datasets/raw/biosbias.json"

if not processed_json_dataset_file.is_file():
    bib._reformat_bias_in_bios_file(
        pkl_file=Path(os.environ['BIOS_PKL']),
        data_dir=Path(COREF_ROOT) / "coref/datasets/raw/",
    )

# ds is a huggingface dataset
ds = bib._load_bias_in_bios(processed_json_dataset_file)["train"]

genders = ["M", "F"]
gendered_ds = {
    gender: [x for x in ds if x["source"]["gender"] == gender] for gender in genders
}


class Vocab(BaseVocab):
    type = "BIOSBIAS"

    def __init__(self, tokenizer_type, split=None):
        assert split is None or split == "train" or split == "test"
        super().__init__(tokenizer_type)
        self.gender = genders[1]  # arbitrary fix a gender
        self.dataset = gendered_ds[self.gender]
        if split == "train":
            self.dataset, _ = train_test_split(self.dataset, 0.7)
        elif split == "test":
            _, self.dataset = train_test_split(self.dataset, 0.7)

        unique_names = list(set(d["entity"] for d in self.dataset))
        unique_attrs = list(set(d["target_mediated"] for d in self.dataset))
        self.filtered_names = self.filter_names(unique_names)
        self.filtered_answers = self.filter_names(unique_attrs)
        self.filtered_attrs = {x: [] for x in self.filtered_answers}
        self.filtered_country_capital_pairs = []
        for d in self.dataset:
            if d["target_mediated"] in self.filtered_answers:
                self.filtered_attrs[d["target_mediated"]].append(
                    len(self.filtered_country_capital_pairs)
                )
                self.filtered_country_capital_pairs.append(
                    (d["attribute"], d["target_mediated"])
                )

    def get_shuffled_labels(self, prompt_id=None):
        rng = self.get_rng(prompt_id)
        names_perm, ans_perm = self.generate_permutations(
            prompt_id, [len(self.filtered_names), len(self.filtered_attrs)]
        )
        attr_rngs = rng.spawn(len(self.filtered_attrs))
        attrs_perm = []
        for attr_rng, ans in zip(attr_rngs, ans_perm):
            pointer = attr_rng.choice(self.filtered_attrs[self.filtered_answers[ans]])
            attrs_perm.append(pointer)
        return names_perm, attrs_perm


class TEMPLATES:
    @classmethod
    def lookup(cls, template_type):
        if template_type == "normal":
            return cls.DEFAULT
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

Question: What occupation does {qn_subject} have?

Answer: {qn_subject} has the occupation of"""
        context_template = """\nAbout {subject}: {country}"""

        def format_parallel(things):
            return ", ".join(things[:-1]) + " and " + things[-1]
