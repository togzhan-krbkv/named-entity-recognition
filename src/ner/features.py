"""Hand-crafted token features for the CRF baseline.

Features are built from word shape and a small context window rather
than pretrained embeddings, since the CRF baseline exists to show what
a classical, non-neural approach can and cannot do on noisy text
before comparing it against a fine-tuned transformer.
"""

import re

from ner.data import TaggedSentence

URL_PATTERN = re.compile(r"^https?://|^www\.")
MENTION_PATTERN = re.compile(r"^@\w+")
HASHTAG_PATTERN = re.compile(r"^#\w+")


def word_shape(word: str) -> dict:
    """Character-level shape features for a single word."""
    return {
        "word.lower": word.lower(),
        "word.suffix3": word[-3:],
        "word.suffix2": word[-2:],
        "word.prefix2": word[:2],
        "word.length": len(word),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_digit": any(c.isdigit() for c in word),
        "word.has_hyphen": "-" in word,
        "word.is_url": bool(URL_PATTERN.match(word)),
        "word.is_mention": bool(MENTION_PATTERN.match(word)),
        "word.is_hashtag": bool(HASHTAG_PATTERN.match(word)),
    }


def token_features(tokens: list[str], index: int) -> dict:
    """Feature dict for tokens[index], including a +/-1 context window."""
    word = tokens[index]
    features = {"bias": 1.0}
    features.update({f"cur.{k}": v for k, v in word_shape(word).items()})

    if index > 0:
        features.update({f"prev.{k}": v for k, v in word_shape(tokens[index - 1]).items()})
    else:
        features["BOS"] = True

    if index < len(tokens) - 1:
        features.update({f"next.{k}": v for k, v in word_shape(tokens[index + 1]).items()})
    else:
        features["EOS"] = True

    return features


def sentence_to_features(sentence: TaggedSentence) -> list[dict]:
    return [token_features(sentence.tokens, i) for i in range(len(sentence.tokens))]


def sentence_to_labels(sentence: TaggedSentence) -> list[str]:
    return list(sentence.tags)
