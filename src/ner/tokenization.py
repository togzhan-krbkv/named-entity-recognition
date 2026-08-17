"""Alignment between word-level BIO labels and subword token sequences.

Fine-tuned transformers operate on subword tokens, but WNUT-17 labels
are per whole word. These functions convert between the two using the
word_ids() list that any fast tokenizer produces, so the alignment
logic itself has no dependency on which tokenizer or model is used and
can be tested without downloading pretrained weights.
"""

IGNORE_INDEX = -100


def align_labels_with_word_ids(
    word_ids: list[int | None],
    word_labels: list[str],
    label2id: dict[str, int],
) -> list[int]:
    """Assign a label id to each subword token.

    Special tokens (word_id is None) get IGNORE_INDEX so the loss
    ignores them. Only the first subword of a word carries the real
    label; later subwords of the same word also get IGNORE_INDEX,
    which is the standard choice for token classification and avoids
    having to invent an I- tag for a continuation that was not
    actually annotated as one.
    """
    aligned: list[int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            aligned.append(label2id[word_labels[word_id]])
        else:
            aligned.append(IGNORE_INDEX)
        previous_word_id = word_id

    return aligned


def align_predictions_to_words(
    word_ids: list[int | None],
    predicted_ids: list[int],
    id2label: dict[int, str],
) -> list[str]:
    """Reduce subword-level predictions to one label per original word.

    Keeps the prediction at each word's first subword, mirroring how
    align_labels_with_word_ids places the real label there. Special
    tokens and continuation subwords are skipped.
    """
    predictions: list[str] = []
    previous_word_id: int | None = None

    for word_id, predicted_id in zip(word_ids, predicted_ids):
        if word_id is None:
            previous_word_id = word_id
            continue
        if word_id != previous_word_id:
            predictions.append(id2label[predicted_id])
        previous_word_id = word_id

    return predictions
