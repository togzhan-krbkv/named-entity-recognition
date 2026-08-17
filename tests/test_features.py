from ner.data import TaggedSentence
from ner.features import sentence_to_features, sentence_to_labels, token_features, word_shape


def test_word_shape_flags_uppercase():
    shape = word_shape("NASA")
    assert shape["word.isupper"] is True
    assert shape["word.istitle"] is False


def test_word_shape_flags_title_case():
    shape = word_shape("London")
    assert shape["word.istitle"] is True
    assert shape["word.isupper"] is False


def test_word_shape_detects_hashtag():
    assert word_shape("#throwback")["word.is_hashtag"] is True
    assert word_shape("throwback")["word.is_hashtag"] is False


def test_word_shape_detects_mention():
    assert word_shape("@paulwalk")["word.is_mention"] is True


def test_word_shape_detects_url():
    assert word_shape("http://example.com")["word.is_url"] is True
    assert word_shape("www.example.com")["word.is_url"] is True
    assert word_shape("example.com")["word.is_url"] is False


def test_word_shape_detects_digits():
    assert word_shape("2017")["word.isdigit"] is True
    assert word_shape("ESB2")["word.has_digit"] is True
    assert word_shape("ESB2")["word.isdigit"] is False


def test_token_features_marks_beginning_of_sentence():
    tokens = ["Hello", "world"]
    features = token_features(tokens, 0)
    assert features["BOS"] is True
    assert "prev.word.lower" not in features


def test_token_features_marks_end_of_sentence():
    tokens = ["Hello", "world"]
    features = token_features(tokens, 1)
    assert features["EOS"] is True
    assert "next.word.lower" not in features


def test_token_features_includes_context_window():
    tokens = ["the", "Empire", "State"]
    features = token_features(tokens, 1)
    assert features["prev.word.lower"] == "the"
    assert features["next.word.lower"] == "state"
    assert features["cur.word.lower"] == "empire"


def test_sentence_to_features_matches_token_count():
    sentence = TaggedSentence(["a", "b", "c"], ["O", "O", "O"])
    features = sentence_to_features(sentence)
    assert len(features) == 3


def test_sentence_to_labels_returns_tags():
    sentence = TaggedSentence(["a", "b"], ["O", "B-person"])
    assert sentence_to_labels(sentence) == ["O", "B-person"]
