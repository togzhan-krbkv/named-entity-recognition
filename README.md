# named-entity-recognition

Named entity recognition on noisy, informal text: baseline CRF and
fine-tuned transformer models, compared at the entity level, served
through a REST API.

Dataset: [WNUT-17](https://noisy-text.github.io/2017/emerging-rare-entities.html)
(Emerging and Rare Entity Recognition): text from Twitter, Stack
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

## Baseline: CRF

`src/ner/features.py` builds hand-crafted per-token features from word
shape and a +/-1 context window: lowercased word, prefix/suffix,
casing, digit and hyphen presence, and simple pattern checks for URLs,
@mentions, and #hashtags. No word embeddings or gazetteers are used,
so the baseline reflects what word-shape features alone can do.

`src/ner/crf_model.py` wraps training and entity-level evaluation
(`sklearn-crfsuite` for the CRF, `seqeval` for metrics). `scripts/train_crf.py`
runs training on the train split, evaluates on dev, and writes the
report to `reports/crf_baseline_eval.md`.

Dev-split result: entity-level micro F1 of 0.183, versus 0.989 on the
training data the model was fit on. That gap is the main finding of
this baseline, not a bug: a small regularization sweep (`c1`, `c2`
from 0.1 up to 3.0) made the train/dev gap worse, not better, which
rules out simple overfitting as the cause. The features are built
largely from word identity (lowercased word, prefix, suffix), and
WNUT-17 is deliberately constructed so that entity surface forms
rarely repeat across splits. A model that leans on word identity has
little to transfer. This motivates the transformer baseline in the
next milestone, which uses subword and contextual representations
instead of memorized word forms.

Per-type F1 on dev is uneven: `location` (0.291) and `person` (0.253)
are the strongest, `corporation` (0.051) and `creative-work` (0.018)
are weak, and `product` and `group` are not detected at all. This
lines up with the entity type distribution from the EDA: the weakest
categories are also the rarest in training data.

## Transformer baseline

`src/ner/tokenization.py` aligns word-level BIO labels to subword
tokens using a fast tokenizer's `word_ids()`: only the first subword
of a word carries the real label, later subwords and special tokens
are masked out of the loss. `src/ner/transformer_model.py` wraps
dataset construction, training (via the Hugging Face `Trainer`), and
entity-level evaluation.

`scripts/train_transformer.py` fine-tunes `distilbert-base-cased`
(chosen over `bert-base-cased` for a faster CPU fine-tuning loop) on
the train split and evaluates on dev with the same entity-level
metric used for the CRF baseline, so the two are directly comparable.

Run it with:

```
python scripts/train_transformer.py
```

This downloads the pretrained weights from Hugging Face Hub on first
run, then fine-tunes for 3 epochs and writes the report to
`reports/transformer_eval.md`.

The alignment logic (`tests/test_tokenization.py`) and the full
training and evaluation pipeline (`tests/test_transformer_pipeline_smoke.py`)
are both verified against a tiny, randomly initialized BERT and a
locally built tokenizer, so the pipeline is proven correct without
requiring network access.

Fine-tuning result (dev split, 3 epochs): entity-level micro F1 of
0.513, up from 0.183 for the CRF baseline. Train loss fell steadily
across epochs (0.236 to 0.048) while dev loss flattened and ticked up
slightly by epoch 3 (0.238 to 0.244), a mild early sign of
overfitting that did not yet hurt dev F1 within 3 epochs.

The gap over the CRF baseline supports the hypothesis from that
milestone: subword and contextual representations transfer to unseen
entity surface forms far better than word-identity features do.
`group` and `product`, which the CRF missed entirely (F1 of 0.000),
are now detected (F1 of 0.159 and 0.272). `person` and `location`
remain the strongest categories. `creative-work` is still the weakest
(F1 of 0.093), consistent with it being both rare in training data and
the hardest category in the original WNUT-17 shared task results.

## Development history

**Data pipeline.** Added a CoNLL parser and BIO-to-span converter
(`src/ner/data.py`), a download script for the raw WNUT-17 files, and
an exploratory analysis script covering split sizes, tag imbalance,
entity type distribution, and entity length distribution.

**CRF baseline.** Added hand-crafted word-shape features, a CRF
training and evaluation pipeline with entity-level metrics, and a
short regularization sweep to check whether the large train/dev gap
was overfitting (it was not).

**Transformer baseline.** Added subword label alignment, a
`distilbert-base-cased` fine-tuning pipeline built on the Hugging Face
`Trainer`, and a smoke test that exercises the full pipeline end to
end on a tiny local model. Fine-tuning raised entity-level F1 from
0.183 (CRF) to 0.513 on the dev split.
