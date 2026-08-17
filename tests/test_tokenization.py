from ner.tokenization import IGNORE_INDEX, align_labels_with_word_ids, align_predictions_to_words

LABEL2ID = {"O": 0, "B-person": 1, "I-person": 2, "B-location": 3, "I-location": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def test_align_labels_ignores_special_tokens():
    # [CLS] word0 word1 [SEP] -> word_ids [None, 0, 1, None]
    word_ids = [None, 0, 1, None]
    labels = ["B-person", "O"]

    aligned = align_labels_with_word_ids(word_ids, labels, LABEL2ID)

    assert aligned == [IGNORE_INDEX, LABEL2ID["B-person"], LABEL2ID["O"], IGNORE_INDEX]


def test_align_labels_only_first_subword_gets_real_label():
    # word 0 splits into two subword tokens: word_ids [0, 0, 1]
    word_ids = [0, 0, 1]
    labels = ["B-location", "O"]

    aligned = align_labels_with_word_ids(word_ids, labels, LABEL2ID)

    assert aligned == [LABEL2ID["B-location"], IGNORE_INDEX, LABEL2ID["O"]]


def test_align_labels_handles_every_word_split_into_subwords():
    # two words, each split into two subwords: word_ids [0, 0, 1, 1]
    word_ids = [0, 0, 1, 1]
    labels = ["B-person", "I-person"]

    aligned = align_labels_with_word_ids(word_ids, labels, LABEL2ID)

    assert aligned == [
        LABEL2ID["B-person"],
        IGNORE_INDEX,
        LABEL2ID["I-person"],
        IGNORE_INDEX,
    ]


def test_align_labels_empty_sentence_only_special_tokens():
    word_ids = [None, None]
    labels: list[str] = []

    aligned = align_labels_with_word_ids(word_ids, labels, LABEL2ID)

    assert aligned == [IGNORE_INDEX, IGNORE_INDEX]


def test_align_predictions_recovers_one_label_per_word():
    word_ids = [None, 0, 0, 1, None]
    predicted_ids = [
        LABEL2ID["O"],  # [CLS], ignored
        LABEL2ID["B-location"],  # word 0, first subword: kept
        LABEL2ID["I-location"],  # word 0, continuation: skipped
        LABEL2ID["O"],  # word 1
        LABEL2ID["O"],  # [SEP], ignored
    ]

    predictions = align_predictions_to_words(word_ids, predicted_ids, ID2LABEL)

    assert predictions == ["B-location", "O"]


def test_align_predictions_matches_original_word_count():
    word_ids = [None, 0, 1, 1, 1, 2, None]
    predicted_ids = [0, 1, 3, 4, 4, 0, 0]

    predictions = align_predictions_to_words(word_ids, predicted_ids, ID2LABEL)

    assert len(predictions) == 3


def test_alignment_round_trips_through_a_multi_subword_sentence():
    # "Alice saw Berlin" where "Berlin" splits into two subwords.
    word_ids = [None, 0, 1, 2, 2, None]
    labels = ["B-person", "O", "B-location"]

    label_ids = align_labels_with_word_ids(word_ids, labels, LABEL2ID)
    recovered = align_predictions_to_words(word_ids, label_ids, ID2LABEL)

    # IGNORE_INDEX positions never get read back since they are never
    # the first subword of a word, so this recovers the original labels.
    assert recovered == labels
