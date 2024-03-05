"""
MCQ abstract dataset.
Can be loaded with arbitrary dataset
"""
from typing import Dict, Tuple, Sequence, List
from dataclasses import dataclass

from coref.datasets.common import (
    BaseVocab,
    MCQStatement,
    Substring,
)


@dataclass
class MCQ:
    question: str
    options: List[str]
    answer: int


class MCQDataset:
    @classmethod
    def num_options(cls):
        raise NotImplementedError

    def get_qn(self, answer, key):
        """
        Returns an MCQ instance
        """
        raise NotImplementedError

    def get_length(self, answer):
        raise NotImplementedError


class MCQVocab(BaseVocab):
    def __init__(self, tokenizer_type, mcq_dataset, labels):
        super().__init__(tokenizer_type)
        self.dataset = mcq_dataset
        self.options = list(range(mcq_dataset.num_options()))
        self.labels = [(label, label) for label in labels]
        for qn_label, ans_label in self.labels:
            assert self.get_length_of_word(qn_label) == 1
            assert self.get_length_of_word(ans_label) == 1

    def get_shuffled_labels(self, prompt_id):
        rng = self.get_rng(prompt_id)
        return rng.spawn(1) + self.generate_permutations(
            prompt_id,
            [
                len(self.options),
                len(self.labels),
            ],
        )

    def fetch_shuffled_name(self, name_id, prompt_id):
        raise NotImplementedError

    def fetch_shuffled_attr(self, attr_id, prompt_id):
        qn_rng, opt_perm, label_perm = self.get_shuffled_labels(prompt_id)
        return self.labels[label_perm[attr_id]]

    def default_context(self, num_entities, labels=None, options=None):
        assert num_entities == len(self.options)
        if labels is None:
            labels = range(num_entities)
        if options is None:
            options = range(num_entities)
        return [MCQStatement(l, o, "normal") for l, o in zip(labels, options)]
