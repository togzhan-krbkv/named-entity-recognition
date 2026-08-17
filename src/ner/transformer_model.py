"""Reusable logic for fine-tuning a token classification transformer.

Kept separate from the download-and-run script so the data
preparation and evaluation logic can be smoke-tested against a tiny
locally built model and tokenizer, without requiring network access
to fetch pretrained weights.
"""

from dataclasses import dataclass

import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast, Trainer, TrainingArguments

from ner.data import TaggedSentence
from ner.tokenization import align_labels_with_word_ids, align_predictions_to_words


def build_label_mappings(sentences: list[TaggedSentence]) -> tuple[dict[str, int], dict[int, str]]:
    """Build label2id/id2label from the tags actually present in the data.

    O is fixed to id 0 so it lines up across any subset of splits that
    happen to be missing a rare entity type.
    """
    labels = {"O"}
    for sentence in sentences:
        labels.update(sentence.tags)

    ordered = ["O"] + sorted(labels - {"O"})
    label2id = {label: i for i, label in enumerate(ordered)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


class NerDataset(Dataset):
    """Tokenizes sentences and aligns labels to subwords on construction."""

    def __init__(
        self,
        sentences: list[TaggedSentence],
        tokenizer: PreTrainedTokenizerFast,
        label2id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        self.encodings = tokenizer(
            [s.tokens for s in sentences],
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_token_type_ids=False,
        )
        self.labels = [
            align_labels_with_word_ids(self.encodings.word_ids(i), sentence.tags, label2id)
            for i, sentence in enumerate(sentences)
        ]
        self.word_ids = [self.encodings.word_ids(i) for i in range(len(sentences))]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


@dataclass
class EvalResult:
    report: str
    f1: float


def evaluate_transformer(
    model: PreTrainedModel,
    dataset: NerDataset,
    id2label: dict[int, str],
) -> EvalResult:
    """Run the model over a dataset and compute entity-level metrics."""
    model.eval()
    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            input_ids = item["input_ids"].unsqueeze(0)
            attention_mask = item["attention_mask"].unsqueeze(0)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_ids = logits.argmax(dim=-1).squeeze(0).tolist()

            word_ids = dataset.word_ids[index]
            true_label_ids = item["labels"].tolist()

            all_pred.append(align_predictions_to_words(word_ids, predicted_ids, id2label))
            all_true.append(align_predictions_to_words(word_ids, true_label_ids, id2label))

    report = classification_report(all_true, all_pred, digits=3)
    f1 = f1_score(all_true, all_pred)
    return EvalResult(report=report, f1=f1)


def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    train_dataset: NerDataset,
    eval_dataset: NerDataset,
    output_dir: str,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 16,
) -> Trainer:
    """Fine-tune model with the Hugging Face Trainer and return it."""
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    return trainer
