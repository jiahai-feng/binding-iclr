"""
From https://github.com/evandez/REMEDI/blob/main/remedi/data.py#L489

MIT License

Copyright (c) 2022 Evan Hernandez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from pathlib import Path
import pickle
from tqdm import tqdm
import logging
import spacy
import json
from typing import Any, Sequence, TypedDict, cast
import datasets

logger = logging.getLogger(__name__)

_BIOS_BIAS_BLACKLISTED_NAMES = frozenset(
    {
        "Non-Residential",
    }
)

# These prefixes do not make as much sense when put in front of the first name, so
# we'll try to remove them as much as possible.
_BIOS_BIAS_PREFIXES = (
    "professor",
    "prof.",
    "prof",
    "dr.",
    "dr",
    "doctor",
    "mr.",
    "mr",
    "ms.",
    "ms",
    "mrs.",
    "mrs",
    "rev.",
    "rev",
    "pastor",
)


class ContextMediationSample(TypedDict):
    """Single sample that can be used for context mediation analysis."""

    id: str  # Identifier
    entity: str  # "Barack Obama"
    attribute: str  # "invented the iPhone"
    context: str  # "Everyone knows that Barack Obama invented the iPhone."
    prompt: str  # "Barack Obama received a degree in"
    target_mediated: str | None  # "computer science" or not set for generation
    target_unmediated: str | None  # "law" or not set for generation
    source: dict | None  # Where this sample was derived from, e.g. counterfact sample.


def load_spacy_model(name: str) -> spacy.language.Language:
    """Load (and cache) a spacy model."""
    return spacy.load(name)


def _reformat_bias_in_bios_file(
    pkl_file: Path,
    data_dir: Path,
    bio_min_words: int = 10,
    sent_min_words: int = 3,
    limit: int | None = 50000,
) -> Path:
    """Reformat the Bias in Bios pickle file on disk."""
    with pkl_file.open("rb") as handle:
        data = pickle.load(handle)

    if limit is not None:
        data = data[:limit]

    # We'll use first name as the entity.
    bb_names = [sample["name"][0] for sample in data]

    # Take only one sentence of each bio to make the task harder.
    nlp = load_spacy_model("en_core_web_sm")
    bb_bios_raw = [sample["raw"] for sample in data]

    bb_bios_abridged = []
    for doc in tqdm(nlp.pipe(bb_bios_raw), total=len(data), desc="parse biosbias"):
        sents = [
            str(sent)
            for sent in doc.sents
            if len(str(sent).strip().split()) >= sent_min_words
        ]
        if len(sents) == 0:
            bb_bio_abridged = ""  # will get filtered out later
        elif len(sents) == 1:
            bb_bio_abridged = sents[0]  # no choice but first
        else:
            bb_bio_abridged = sents[1]  # otherwise, always take second sentence
        bb_bios_abridged.append(bb_bio_abridged)

    # Normalize the samples.
    lines = []
    for index, (sample, bb_name, bb_bio) in enumerate(
        zip(data, bb_names, bb_bios_abridged)
    ):
        if bb_name in _BIOS_BIAS_BLACKLISTED_NAMES:
            logger.debug(
                f"will not include sample #{index} because it has "
                f"blacklisted name '{bb_name}'"
            )
            continue

        # Replace all variants of the person's name with their first name.
        # We'll ignore suffixes, but try to strip prefixes since they are much more
        # common and do not mesh well with just first names.
        full_name = " ".join(part for part in sample["name"] if part)
        last_name = sample["name"][-1]
        first_last_name = f"{bb_name} {last_name}"

        replacements = []
        for name in (full_name, first_last_name, last_name):
            for prefix in _BIOS_BIAS_PREFIXES:
                replacements.append(f"{prefix} {name}")
                replacements.append(f"{prefix.capitalize()} {name}")
                replacements.append(f"{prefix.upper()} {name}")
        replacements += [full_name, first_last_name, last_name]
        for replace in replacements:
            bb_bio = bb_bio.replace(replace, bb_name)
        bb_bio = bb_bio.strip("*• ")

        bb_title = sample["title"].replace("_", " ")
        bb_id = "_".join(part for part in sample["name"] if part)

        # NOTE(evan): This filtering is necessary due to a downstream limitation where
        # the editor assumes that the last occurrence of an entity is always the second
        # occurrence.
        n_occurrences = bb_bio.count(bb_name)
        if n_occurrences == 0:
            bb_bio = f"About {bb_name}: {bb_bio}"
        elif n_occurrences != 1:
            logger.debug(
                f"will not include sample #{index} because there are "
                f"{n_occurrences} (!= 1) occurrences of '{bb_name}' in '{bb_bio}'"
            )
            continue

        approx_n_words = len(bb_bio.split())
        if approx_n_words < bio_min_words:
            logger.debug(
                f"will not include sample #{index} because bio '{bb_bio}' contains "
                f"too few words (approx. < {bio_min_words})"
            )
            continue

        entity = bb_name
        prompt = f"{entity} has the occupation of"
        context = bb_bio
        attribute = bb_bio[bb_bio.index(bb_name) + len(bb_name) :].strip(".: ")
        target_mediated = bb_title

        line = ContextMediationSample(
            id=bb_id,
            source=sample,
            entity=entity,
            prompt=prompt,
            context=context,
            attribute=attribute,
            target_mediated=target_mediated,
            target_unmediated=None,
        )
        lines.append(line)

    # Save in jsonl format.
    json_file = data_dir / "biosbias.json"
    with json_file.open("w") as handle:
        for line in lines:
            json.dump(dict(line), handle)
    return json_file


def _load_bias_in_bios(file: Path, **kwargs: Any):
    """Load the Bias in Bios datast, if possible."""

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(
            f"biosbias file not found: {file}"
            "\nnote this dataset cannot be downloaded automatically, "
            "so you will have to retrieve it by following the instructions "
            "at the github repo https://github.com/microsoft/biosbias and pass "
            "the .pkl file in via the -f flag"
        )

    if file.suffix in {".pkl", ".pickle"}:
        logger.debug(f"{file} is pickle format; will reformat into json")
        file = _reformat_bias_in_bios_file(file)

    dataset = datasets.load_dataset("json", data_files=str(file), **kwargs)
    assert isinstance(
        dataset, datasets.arrow_dataset.Dataset | datasets.dataset_dict.DatasetDict
    ), type(dataset)
    return dataset
