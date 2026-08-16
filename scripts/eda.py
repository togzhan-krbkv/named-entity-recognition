"""Exploratory analysis of the WNUT-17 splits.

Writes a markdown summary to reports/eda.md and a bar chart of entity
type frequency to reports/entity_type_distribution.png.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from ner.data import extract_entities, load_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def summarize_split(split: str) -> dict:
    sentences = load_split(DATA_DIR, split)
    lengths = [len(s) for s in sentences]

    entity_type_counts: Counter = Counter()
    entity_length_counts: Counter = Counter()
    tag_counts: Counter = Counter()

    for sentence in sentences:
        tag_counts.update(sentence.tags)
        for entity in extract_entities(sentence.tags):
            entity_type_counts[entity.entity_type] += 1
            entity_length_counts[entity.end - entity.start] += 1

    non_o_tags = sum(count for tag, count in tag_counts.items() if tag != "O")

    return {
        "split": split,
        "sentences": len(sentences),
        "tokens": sum(lengths),
        "avg_sentence_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_sentence_length": max(lengths) if lengths else 0,
        "entities": sum(entity_type_counts.values()),
        "entity_type_counts": entity_type_counts,
        "entity_length_counts": entity_length_counts,
        "o_tag_fraction": tag_counts["O"] / sum(tag_counts.values()) if tag_counts else 0,
        "non_o_tags": non_o_tags,
    }


def write_report(summaries: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# WNUT-17 exploratory analysis", ""]

    lines.append("## Split sizes")
    lines.append("")
    lines.append("| split | sentences | tokens | avg length | max length | entities | O-tag fraction |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['split']} | {s['sentences']} | {s['tokens']} | "
            f"{s['avg_sentence_length']:.1f} | {s['max_sentence_length']} | "
            f"{s['entities']} | {s['o_tag_fraction']:.3f} |"
        )
    lines.append("")

    lines.append("## Entity type distribution (train)")
    lines.append("")
    lines.append("| entity type | count |")
    lines.append("|---|---|")
    train_types = summaries[0]["entity_type_counts"]
    for entity_type, count in train_types.most_common():
        lines.append(f"| {entity_type} | {count} |")
    lines.append("")

    lines.append("## Entity length distribution (train, in tokens)")
    lines.append("")
    lines.append("| length | count |")
    lines.append("|---|---|")
    train_lengths = summaries[0]["entity_length_counts"]
    for length in sorted(train_lengths):
        lines.append(f"| {length} | {train_lengths[length]} |")
    lines.append("")

    (REPORTS_DIR / "eda.md").write_text("\n".join(lines), encoding="utf-8")


def write_plot(train_summary: dict) -> None:
    counts = train_summary["entity_type_counts"].most_common()
    labels = [c[0] for c in counts]
    values = [c[1] for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylabel("count")
    ax.set_title("Entity type distribution (train)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "entity_type_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    summaries = [summarize_split(split) for split in ("train", "dev", "test")]
    write_report(summaries)
    write_plot(summaries[0])


if __name__ == "__main__":
    main()
