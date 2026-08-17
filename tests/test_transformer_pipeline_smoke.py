"""Smoke test for the transformer training pipeline.

Uses a tiny, randomly initialized BERT and a tokenizer built from a
local vocab file, so it runs offline and in seconds. It does not
check accuracy (a random tiny model on four sentences learns nothing
meaningful) — it exists to catch plumbing bugs (shape mismatches,
API drift in the Trainer, broken alignment) before running the real
fine-tuning job, which needs downloaded pretrained weights and is not
run in this environment.
"""

import pytest
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from ner.data import TaggedSentence
from ner.transformer_model import NerDataset, build_label_mappings, evaluate_transformer, run_training

SENTENCES = [
    TaggedSentence(["Alice", "went", "to", "Paris"], ["B-person", "O", "O", "B-location"]),
    TaggedSentence(["Bob", "saw", "Alice"], ["B-person", "O", "B-person"]),
    TaggedSentence(["Paris", "is", "nice"], ["B-location", "O", "O"]),
    TaggedSentence(
        ["I", "like", "Paris", "and", "Berlin"],
        ["O", "O", "B-location", "O", "B-location"],
    ),
]


@pytest.fixture
def tiny_tokenizer(tmp_path) -> BertTokenizerFast:
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = sorted({token.lower() for sentence in SENTENCES for token in sentence.tokens})
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(special_tokens + words), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)


def test_training_pipeline_runs_end_to_end(tmp_path, tiny_tokenizer):
    label2id, id2label = build_label_mappings(SENTENCES)

    config = BertConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model = BertForTokenClassification(config)

    train_dataset = NerDataset(SENTENCES, tiny_tokenizer, label2id, max_length=16)
    eval_dataset = NerDataset(SENTENCES, tiny_tokenizer, label2id, max_length=16)

    trainer = run_training(
        model=model,
        tokenizer=tiny_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(tmp_path / "out"),
        num_train_epochs=1,
        batch_size=2,
    )

    result = evaluate_transformer(trainer.model, eval_dataset, id2label)

    assert 0.0 <= result.f1 <= 1.0
    assert "precision" in result.report
