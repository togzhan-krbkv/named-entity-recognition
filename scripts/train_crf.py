"""Train the CRF baseline and evaluate it on the dev split.

Evaluation is entity-level (via seqeval), not token-level, since
token-level accuracy is inflated by the dominant O tag and does not
reflect how well entities are actually detected.
"""

from pathlib import Path

import joblib

from ner.crf_model import evaluate_crf, train_crf
from ner.data import load_split
from ner.features import sentence_to_features, sentence_to_labels

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def build_dataset(split: str, data_dir: Path = DATA_DIR) -> tuple[list[list[dict]], list[list[str]]]:
    sentences = load_split(data_dir, split)
    features = [sentence_to_features(s) for s in sentences]
    labels = [sentence_to_labels(s) for s in sentences]
    return features, labels


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = build_dataset("train")
    crf = train_crf(x_train, y_train)
    joblib.dump(crf, MODELS_DIR / "crf_baseline.joblib")

    x_dev, y_dev = build_dataset("dev")
    dev_report, dev_f1 = evaluate_crf(crf, x_dev, y_dev)

    lines = [
        "# CRF baseline evaluation (dev split)",
        "",
        f"Entity-level micro F1: {dev_f1:.3f}",
        "",
        "```",
        dev_report,
        "```",
    ]
    (REPORTS_DIR / "crf_baseline_eval.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
