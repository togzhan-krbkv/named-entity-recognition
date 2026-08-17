"""Fine-tune a pretrained transformer for token classification on WNUT-17.

Requires network access to download the base model from Hugging Face
Hub. Run after scripts/download_data.py.
"""

from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner.data import load_split
from ner.transformer_model import NerDataset, build_label_mappings, evaluate_transformer, run_training

MODEL_NAME = "distilbert-base-cased"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "transformer"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_sentences = load_split(DATA_DIR, "train")
    dev_sentences = load_split(DATA_DIR, "dev")
    test_sentences = load_split(DATA_DIR, "test")

    label2id, id2label = build_label_mappings(train_sentences + dev_sentences + test_sentences)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = NerDataset(train_sentences, tokenizer, label2id)
    dev_dataset = NerDataset(dev_sentences, tokenizer, label2id)

    trainer = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=str(MODELS_DIR / "checkpoints"),
    )

    result = evaluate_transformer(trainer.model, dev_dataset, id2label)

    trainer.save_model(str(MODELS_DIR / "final"))
    tokenizer.save_pretrained(str(MODELS_DIR / "final"))

    lines = [
        "# Transformer baseline evaluation (dev split)",
        "",
        f"Model: {MODEL_NAME}",
        f"Entity-level micro F1: {result.f1:.3f}",
        "",
        "```",
        result.report,
        "```",
    ]
    (REPORTS_DIR / "transformer_eval.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
