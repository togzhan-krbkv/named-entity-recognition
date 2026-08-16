"""Loading and parsing utilities for the WNUT-17 NER dataset.

The dataset is stored in CoNLL format: one token and its BIO tag per
line, separated by whitespace, with a blank line between sentences.
"""

from dataclasses import dataclass
from pathlib import Path

SPLIT_FILENAMES = {
    "train": "wnut17train.conll",
    "dev": "emerging.dev.conll",
    "test": "emerging.test.annotated",
}


@dataclass
class TaggedSentence:
    """A single sentence as parallel lists of tokens and BIO tags."""

    tokens: list[str]
    tags: list[str]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class Entity:
    """A named entity span within a sentence.

    start is inclusive, end is exclusive, both indices into the
    sentence's token list.
    """

    entity_type: str
    start: int
    end: int

    def text(self, tokens: list[str]) -> str:
        return " ".join(tokens[self.start : self.end])


def parse_conll(path: Path) -> list[TaggedSentence]:
    """Parse a CoNLL-formatted file into a list of tagged sentences.

    Lines with fewer than two fields are skipped, which handles the
    occasional malformed line in the raw WNUT-17 files.
    """
    sentences: list[TaggedSentence] = []
    tokens: list[str] = []
    tags: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if tokens:
                    sentences.append(TaggedSentence(tokens, tags))
                    tokens, tags = [], []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            tokens.append(parts[0])
            tags.append(parts[-1])

    if tokens:
        sentences.append(TaggedSentence(tokens, tags))

    return sentences


def load_split(data_dir: Path, split: str) -> list[TaggedSentence]:
    """Load one of the train, dev, or test splits from data_dir."""
    if split not in SPLIT_FILENAMES:
        raise ValueError(f"unknown split '{split}', expected one of {list(SPLIT_FILENAMES)}")
    return parse_conll(data_dir / SPLIT_FILENAMES[split])


def extract_entities(tags: list[str]) -> list[Entity]:
    """Convert a sequence of BIO tags into entity spans.

    An I- tag that appears without a preceding B- of the same type is
    treated as the start of a new entity, which matches how seqeval
    and most NER evaluation tools handle malformed tag sequences.
    """
    entities: list[Entity] = []
    start: int | None = None
    entity_type: str | None = None

    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                entities.append(Entity(entity_type, start, i))
            start, entity_type = i, tag[2:]
        elif tag.startswith("I-"):
            current_type = tag[2:]
            if start is None or current_type != entity_type:
                if start is not None:
                    entities.append(Entity(entity_type, start, i))
                start, entity_type = i, current_type
        else:
            if start is not None:
                entities.append(Entity(entity_type, start, i))
            start, entity_type = None, None

    return entities
