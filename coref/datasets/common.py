import itertools
import torch
from collections import namedtuple
from coref.utils import prepend
import pandas as pd
import os
import numpy as np
from coref import COREF_ROOT
from coref.models import LLAMA_PATH, HF_PATH
from transformers import AutoTokenizer, LlamaTokenizerFast

print("importing dataset common")


df = pd.read_csv(os.path.join(COREF_ROOT, "coref/datasets/raw/new-top-firstNames.csv"))
names = df.name.to_list()


df = pd.read_csv(os.path.join(COREF_ROOT, "coref/datasets/raw/country-list.csv"))
country_capital_pairs = [(r.country, r.capital) for r in df.itertuples()]


df = pd.read_csv(os.path.join(COREF_ROOT, "coref/datasets/raw/colors.csv"))
colors = df.color.to_list()

from enum import IntEnum


class Name(IntEnum):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5


class Attr(IntEnum):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5


from string import Formatter
from collections import defaultdict


def train_test_split(lst, frac):
    mid = int(len(lst) * frac)
    return lst[:mid], lst[mid:]


class TrackFormatter(Formatter):
    def format(self, format_string, **kwargs):
        """
        Only accepts keyword arguments.

        Returns:
            formatted string : str
            locations : Dict[str, List[Tuple[int, int]]] - denoting start and end positions (inclusive, exclusive)
        """
        locations = defaultdict(list)
        result = []
        run_length = 0
        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):
            # output the literal text
            if literal_text:
                result.append(literal_text)
                run_length += len(literal_text)

            if field_name is not None:
                # given the field_name, find the object it references
                #  and the argument it came from
                obj, arg_used = self.get_field(field_name, [], kwargs)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)

                # format the object and append to the result
                final_str = self.format_field(obj, format_spec)
                locations[arg_used].append(
                    Substring(run_length, run_length + len(final_str))
                )
                result.append(final_str)
                run_length += len(final_str)

        return "".join(result), locations


from dataclasses import dataclass


@dataclass(frozen=True)
class Substring:
    start: int
    end: int

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, key):
        if key == 0:
            return self.start
        else:
            return self.end

    def to_slice(self):
        return slice(self.start, self.end)

    def __add__(self, num):
        return Substring(self.start + num, self.end + num)


import bisect


def recursify(func, dtype=Substring, pred=None):
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        if pred(indices):
            return func(indices, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, *args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, *args, **kwargs) for value in indices]
        else:
            raise Exception(f"Unexpected type {type(indices)}")

    return wrapper


@recursify
def recursive_align_tokens(indices, offset_mapping):
    # it's unclear what conventions offset_mapping uses
    # but we can say for sure that starting indices are inclusive
    # but my indices are inclusive, exclusive
    start, end = indices
    start = bisect.bisect_right([x for x, _ in offset_mapping], start) - 1
    end = bisect.bisect_right([x for x, _ in offset_mapping], end - 1) - 1
    return Substring(start, end + 1)


@recursify
def recursive_add_offset(indices, offset):
    return indices + offset


def rotate(func, dtype=Substring, pred=None):
    """
    Rotates "List ... x" into "... List x", and calls func on List x

    Returns:
        ... func(List x)
    """
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        assert isinstance(indices, list)
        child = indices[0]
        if pred(child):
            return func(indices, *args, **kwargs)
        elif isinstance(child, dict):
            return {
                key: wrapper([sib[key] for sib in indices], *args, **kwargs)
                for key in child.keys()
            }

        elif isinstance(child, list):
            return [
                wrapper([sib[key] for sib in indices], *args, **kwargs)
                for key in range(len(child))
            ]
        else:
            raise Exception(f"Unexpected type {type(child)}")

    return wrapper


class BaseVocab:
    train_prop = 0.5
    def __init__(self, tokenizer_type):
        """
        tokenizer_type: "pythia" or "llama"
        """
        if tokenizer_type == "pythia":
            self.LLAMA = False
        elif tokenizer_type == "llama":
            self.LLAMA = True
        else:
            raise Exception(f"Unknown tokenizer type {tokenizer_type}")
        if self.LLAMA:
            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                os.path.join(LLAMA_PATH, "7B")
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-2.8b",
                **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            )
        self.global_rng = np.random.default_rng(42)

    def filter_names(self, names, length=1):
        num_tokens = [self.get_length_of_word(name) for name in names]
        filtered = [name for name, ln in zip(names, num_tokens) if ln == length]
        return filtered

    def filter_countries(self, country_capital_pairs, length=1):
        num_tokens = [
            self.get_length_of_word(country) for country, _ in country_capital_pairs
        ]
        num_tokens2 = [
            self.get_length_of_word(city) for country, city in country_capital_pairs
        ]
        filtered = [
            pair
            for pair, ln, ln2 in zip(country_capital_pairs, num_tokens, num_tokens2)
            if ln == length and ln2 == length
        ]
        return filtered

    def get_rng(self, prompt_id):
        if prompt_id is not None:
            rng = np.random.default_rng(prompt_id)
        else:
            assert "Prompt_id not provided to vocab!"
            rng = self.global_rng
        return rng

    def generate_permutations(self, prompt_id, lengths):
        rng = self.get_rng(prompt_id)
        return [rng.permutation(l) for l in lengths]

    def get_shuffled_labels(self, prompt_id=None):
        return self.generate_permutations(
            prompt_id,
            [len(self.filtered_names), len(self.filtered_country_capital_pairs)],
        )

    def fetch_shuffled_name(self, name_id, prompt_id):
        names_perm, _ = self.get_shuffled_labels(prompt_id)
        return self.filtered_names[names_perm[name_id]]

    def fetch_shuffled_attr(self, attr_id, prompt_id):
        _, attrs_perm = self.get_shuffled_labels(prompt_id)
        return self.filtered_country_capital_pairs[attrs_perm[attr_id]]

    def encode_single_word(self, word):
        if self.LLAMA:
            stuff = self.tokenizer.encode(word)[1:]
        else:
            stuff = self.tokenizer.encode(" " + word)
        assert len(stuff) == 1
        return stuff[0]

    def get_length_of_word(self, word):
        if self.LLAMA:
            return len(self.tokenizer.encode(word)) - 1
        else:
            return len(self.tokenizer(" " + word)["input_ids"])

    def simple_train_test_split(self, split):
        if split == "train":
            self.filtered_names, _ = train_test_split(self.filtered_names, self.train_prop)
            self.filtered_country_capital_pairs, _ = train_test_split(
                self.filtered_country_capital_pairs, self.train_prop
            )
        elif split == "test":
            _, self.filtered_names = train_test_split(self.filtered_names, self.train_prop)
            _, self.filtered_country_capital_pairs = train_test_split(
                self.filtered_country_capital_pairs, self.train_prop
            )
        else:
            assert split is None

    @staticmethod
    def default_context(num_entities, entities=None, attributes=None):
        if entities is None:
            entities = range(num_entities)
        if attributes is None:
            attributes = range(num_entities)
        return [Statement(e, s, "normal") for e, s in zip(entities, attributes)]


def tokenize_prompt(*, vocab, prompt, indices, **kwargs):
    tokenized = vocab.tokenizer(prompt, return_offsets_mapping=True)
    aligned_tokens = recursive_align_tokens(indices, tokenized["offset_mapping"])
    return tokenized["input_ids"], aligned_tokens


def stack_tokens(vocab, tokens_list):
    longest_length = max(len(s) for s in tokens_list)
    assert type(vocab.tokenizer.pad_token_type_id) == int
    padded = torch.tensor(
        [
            s + [vocab.tokenizer.pad_token_type_id] * (longest_length - len(s))
            for s in tokens_list
        ]
    )
    return padded


default_mcq_labels = ["A", "B", "C", "D", "E", "F", "G"]


def generate_shuffler(prompt_id, num_entities):
    rng = np.random.default_rng(prompt_id)
    perm = rng.permutation(num_entities)
    inv_perm = perm.copy()
    for i, p in enumerate(perm):
        inv_perm[p[i]] = i

    def shuffler(ls, invert=False):
        assert len(ls) == num_entities
        if invert:
            return [ls[p] for p in perm]
        else:
            return [ls[p] for p in inv_perm]

    return shuffler


Statement = namedtuple("Statement", ["name", "attr", "type"])

MCQStatement = namedtuple(
    "MCQStatement", ["label", "option", "type", "raw_label"], defaults=[None]
)

ICLStatement = namedtuple(
    "ICLStatement", ["question", "answer", "label", "type"], defaults=[None]
)
