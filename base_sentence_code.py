from datasets import Dataset
from evaluate import load
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, TrainingArguments, Trainer

import numpy as np
import pandas
import torch.nn as nn
import torch.optim as optim


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return load("accuracy").compute(predictions=predictions, references=labels)


# snapshot_download(repo_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english", repo_type="model", local_dir="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", local_files_only=True)

# hf_hub_download(repo_id="dair-ai/emotion", filename="split/train-00000-of-00001.parquet", repo_type="dataset", local_dir=".")
train_dataframe = pandas.read_parquet('split/train-00000-of-00001.parquet')
train_text = train_dataframe["text"].values.tolist()
train_labels = train_dataframe["label"].values.tolist()
train_dataset = Dataset.from_dict({"text": train_text, "label": train_labels})
tokenized_train = train_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

# hf_hub_download(repo_id="dair-ai/emotion", filename="split/test-00000-of-00001.parquet", repo_type="dataset", local_dir=".")
eval_dataframe = pandas.read_parquet('split/test-00000-of-00001.parquet')
eval_text = eval_dataframe["text"].values.tolist()
eval_labels = eval_dataframe["label"].values.tolist()
eval_dataset = Dataset.from_dict({"text": eval_text, "label": eval_labels})
tokenized_eval = eval_dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())
