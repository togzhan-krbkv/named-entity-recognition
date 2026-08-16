# named-entity-recognition

Named entity recognition on noisy, informal text: baseline CRF and
fine-tuned transformer models, compared at the entity level, served
through a REST API.

Dataset: [WNUT-17](https://noisy-text.github.io/2017/emerging-rare-entities.html)
(Emerging and Rare Entity Recognition) — text from Twitter, Stack
Overflow, YouTube, and Reddit, annotated for six entity types: person,
location, group, corporation, product, and creative-work.

Status: in progress.

## Setup

```
pip install -e .
pip install -r requirements.txt
python scripts/download_data.py
```

## Dataset and exploratory analysis

`src/ner/data.py` parses the CoNLL-formatted files into sentences with
parallel token and BIO-tag lists, and converts tag sequences into
entity spans.

`scripts/eda.py` produces a summary in `reports/eda.md` and a plot of
entity type frequency in `reports/entity_type_distribution.png`.

Key characteristics of the data:

- 3,394 training sentences, 1,009 dev, 1,287 test.
- Roughly 95% of tokens carry the `O` tag; entities are sparse, which
  matters for both model choice and evaluation (entity-level F1 rather
  than token-level accuracy).
- Entity types are imbalanced: `person` and `location` together
  account for over half of all training entities, while `product` and
  `creative-work` are comparatively rare.
- Most entities span 1-2 tokens, with a long tail up to 14 tokens.
- The dataset is deliberately high-variance: entity surface forms
  repeat far less than in standard newswire NER corpora, which is why
  it is used to test generalization to unseen entities rather than
  memorization of frequent ones.

Full split-by-split statistics are in `reports/eda.md`.

## Development history

**Data pipeline.** Added a CoNLL parser and BIO-to-span converter
(`src/ner/data.py`), a download script for the raw WNUT-17 files, and
an exploratory analysis script covering split sizes, tag imbalance,
entity type distribution, and entity length distribution.
