from ner.crf_model import evaluate_crf, train_crf
from ner.data import TaggedSentence
from ner.features import sentence_to_features, sentence_to_labels

# A tiny, deterministic synthetic dataset: the word "Paris" is always
# a location, "Alice" is always a person, everything else is O. A CRF
# trained on repeated examples of this should fit it exactly, which
# makes the test a check on the training/evaluation plumbing rather
# than a check on real-world NER quality.
SENTENCES = [
    TaggedSentence(["Alice", "went", "to", "Paris"], ["B-person", "O", "O", "B-location"]),
    TaggedSentence(["Paris", "is", "nice"], ["B-location", "O", "O"]),
    TaggedSentence(["Alice", "called"], ["B-person", "O"]),
    TaggedSentence(["I", "saw", "Alice", "in", "Paris"], ["O", "O", "B-person", "O", "B-location"]),
]


def build_xy(sentences: list[TaggedSentence]) -> tuple[list[list[dict]], list[list[str]]]:
    x = [sentence_to_features(s) for s in sentences]
    y = [sentence_to_labels(s) for s in sentences]
    return x, y


def test_train_crf_fits_deterministic_pattern():
    x, y = build_xy(SENTENCES)

    crf = train_crf(x, y, c1=0.01, c2=0.01, max_iterations=50)
    predictions = [list(sequence) for sequence in crf.predict(x)]

    assert predictions == y


def test_evaluate_crf_reports_perfect_score_on_fit_data():
    x, y = build_xy(SENTENCES)
    crf = train_crf(x, y, c1=0.01, c2=0.01, max_iterations=50)

    report, f1 = evaluate_crf(crf, x, y)

    assert f1 == 1.0
    assert "location" in report
    assert "person" in report


def test_evaluate_crf_returns_valid_score_on_unseen_sentence():
    x, y = build_xy(SENTENCES)
    crf = train_crf(x, y, c1=0.01, c2=0.01, max_iterations=50)

    unseen = [TaggedSentence(["Bob", "went", "to", "Berlin"], ["B-person", "O", "O", "B-location"])]
    x_unseen, y_unseen = build_xy(unseen)

    _, f1 = evaluate_crf(crf, x_unseen, y_unseen)

    assert 0.0 <= f1 <= 1.0
