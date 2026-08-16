from pathlib import Path

import pytest

from ner.data import Entity, TaggedSentence, extract_entities, load_split, parse_conll


def write_conll(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_conll_splits_sentences_on_blank_lines(tmp_path):
    path = write_conll(
        tmp_path,
        "sample.conll",
        "Empire\tB-location\nState\tI-location\n\nHello\tO\nworld\tO\n",
    )

    sentences = parse_conll(path)

    assert sentences == [
        TaggedSentence(["Empire", "State"], ["B-location", "I-location"]),
        TaggedSentence(["Hello", "world"], ["O", "O"]),
    ]


def test_parse_conll_handles_missing_trailing_blank_line(tmp_path):
    path = write_conll(tmp_path, "sample.conll", "Hi\tO\nthere\tO")

    sentences = parse_conll(path)

    assert sentences == [TaggedSentence(["Hi", "there"], ["O", "O"])]


def test_parse_conll_skips_malformed_lines(tmp_path):
    path = write_conll(tmp_path, "sample.conll", "Hi\tO\n\nthere\tO\n")

    sentences = parse_conll(path)

    assert sentences == [
        TaggedSentence(["Hi"], ["O"]),
        TaggedSentence(["there"], ["O"]),
    ]


def test_parse_conll_empty_file(tmp_path):
    path = write_conll(tmp_path, "sample.conll", "")

    assert parse_conll(path) == []


def test_load_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        load_split(tmp_path, "validation")


def test_load_split_reads_correct_file(tmp_path):
    write_conll(tmp_path, "wnut17train.conll", "Hi\tO\n")

    sentences = load_split(tmp_path, "train")

    assert sentences == [TaggedSentence(["Hi"], ["O"])]


def test_extract_entities_simple_span():
    tags = ["O", "B-location", "I-location", "O"]

    assert extract_entities(tags) == [Entity("location", 1, 3)]


def test_extract_entities_multiple_spans():
    tags = ["B-person", "O", "B-corporation", "I-corporation"]

    assert extract_entities(tags) == [
        Entity("person", 0, 1),
        Entity("corporation", 2, 4),
    ]


def test_extract_entities_no_entities():
    assert extract_entities(["O", "O", "O"]) == []


def test_extract_entities_i_tag_without_preceding_b_tag():
    # Malformed sequence: treat the stray I- tag as starting a new entity.
    tags = ["O", "I-location", "I-location"]

    assert extract_entities(tags) == [Entity("location", 1, 3)]


def test_extract_entities_type_change_without_b_tag():
    # Malformed sequence: an I- tag of a different type starts a new entity.
    tags = ["B-person", "I-corporation"]

    assert extract_entities(tags) == [
        Entity("person", 0, 1),
        Entity("corporation", 1, 2),
    ]


def test_extract_entities_span_at_end_of_sentence():
    tags = ["O", "B-product", "I-product"]

    assert extract_entities(tags) == [Entity("product", 1, 3)]


def test_entity_text_returns_surface_form():
    tokens = ["the", "Empire", "State", "Building"]
    entity = Entity("location", 1, 4)

    assert entity.text(tokens) == "Empire State Building"
