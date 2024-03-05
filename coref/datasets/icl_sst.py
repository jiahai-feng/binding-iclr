import pandas as pd
import os
from coref import COREF_ROOT
import itertools
from coref.datasets.common import colors, TrackFormatter, Substring
from coref.datasets.overthinking_datasets import (
    get_dataset,
    SST2_DATASET_PARAMS,
    SST2_LABELS,
)
from coref.datasets.mcq import MCQDataset, MCQ, MCQVocab


class Vocab(MCQVocab):
    type = "ICL_SST"

    def __init__(self, tokenizer_type):
        super().__init__(
            tokenizer_type, SSTDataset(), ["red", "blue", "green", "white", "black"]
        )


class SSTDataset(MCQDataset):
    @classmethod
    def num_options(cls):
        assert 2 == len(SST2_LABELS[0])
        return len(SST2_LABELS[0])

    def get_qn(self, answer, key):
        """
        Returns an MCQ instance
        """
        return self.dataset[answer][key]

    def get_length(self, answer):
        return len(self.dataset[answer])

    def __init__(self):
        dataset = get_dataset(SST2_DATASET_PARAMS)
        assert len(dataset) == len(SST2_LABELS[0])
        # Flatten and process
        self.dataset = [
            [
                MCQ(
                    question=question[0],
                    options=SST2_LABELS[0],
                    answer=answer,
                )
                for question in subset
            ]
            for answer, subset in enumerate(dataset)
        ]


class TEMPLATES:
    @classmethod
    def lookup(cls, template_type):
        if template_type == "normal":
            return cls.DEFAULT
        raise Exception(f"unknown template {template_type}")

    class DEFAULT:
        """
        Implicitly set pseudo-label = pseudo-option
        """

        @classmethod
        def lookup(cls, context_type):
            if context_type == "normal":
                return cls.context_template
            raise Exception(f"unknown context template {context_type}")

        @classmethod
        def lookup_qn(cls, vocab, qn_rng, opt_perm, query_name, raw_query_name, nth):
            assert raw_query_name is None, "Raw query name not supported"
            real_query = opt_perm[query_name]
            for _ in range(nth + 1):
                qn_id = qn_rng.integers(vocab.dataset.get_length(real_query))
            return vocab.dataset.get_qn(real_query, qn_id)

        @classmethod
        def generate_template(cls, *, vocab, template_context, prompt_id, context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name):
                assert prompt_id is not None
                qn_rng, opt_perm, label_perm = vocab.get_shuffled_labels(prompt_id)
                # pick first qn in the permutation
                cur_qn = cls.lookup_qn(
                    vocab=vocab,
                    qn_rng=qn_rng,
                    opt_perm=opt_perm,
                    query_name=query_name,
                    raw_query_name=raw_query_name,
                    nth=0,
                )
                return cls.template, {"question": cur_qn.question}

            return ret

        @classmethod
        def instantiate(cls, vocab, statement, prompt_id, template_context):
            @lambda f: f(**template_context)
            def ret(query_name, raw_query_name):
                context_template = cls.lookup(statement.type)
                qn_rng, opt_perm, label_perm = vocab.get_shuffled_labels(prompt_id)
                qn = cls.lookup_qn(
                    vocab=vocab,
                    qn_rng=qn_rng,
                    opt_perm=opt_perm,
                    query_name=statement.answer,
                    raw_query_name=raw_query_name,
                    nth=statement.question + 1,
                )
                label, answer_label = vocab.labels[label_perm[statement.label]]
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    context_template,
                    question=qn.question,
                    answer=label,
                )
                return (
                    cur_ctx,
                    {
                        "question": ctx_idx_map["answer"][0],
                        "answer": ctx_idx_map["answer"][0],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )

            return ret

        @classmethod
        def extract_template_indices(cls, full_output_indices):
            return {}

        template = """{context}\n\nReview: {question}\nAnswer:"""  # SST prompt 1
        context_template = """\n\nReview: {question}\nAnswer: {answer}"""
