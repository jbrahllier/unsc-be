#!/usr/bin/env python
"""
actor_classification.py

Multi-label classify the actors in some number of clauses.
"""

import argparse
import os
import json
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
from datetime import datetime

# custom binary cross entropy loss trainer
class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

# metrics being tracked
accuracy_metric  = evaluate.load("accuracy")
f1_metric        = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")

def multi_label_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits))
    preds = (probs >= 0.5).int().numpy()
    flat_preds = preds.reshape(-1)
    flat_labels = labels.reshape(-1)
    return {
        "accuracy":  accuracy_metric.compute(predictions=flat_preds, references=flat_labels)["accuracy"],
        "f1":        f1_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["f1"],
        "precision": precision_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["precision"],
        "recall":    recall_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["recall"],
    }

# sliding window tokenizer
def chunk_and_tokenize(examples, tokenizer, max_length=512, stride=256):
    all_input_ids, all_attention_masks, all_labels = [], [], []
    for clause, label_vec in zip(examples["clause"], examples["labels"]):
        tokens = tokenizer.encode(clause, add_special_tokens=True)
        for start in range(0, len(tokens), stride):
            window = tokens[start:start+max_length]
            mask = [1]*len(window)
            pad_len = max_length - len(window)
            if pad_len > 0:
                window += [tokenizer.pad_token_id]*pad_len
                mask += [0]*pad_len
            all_input_ids.append(window)
            all_attention_masks.append(mask)
            all_labels.append(label_vec)
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }

# global variable declarations
BASE_MODEL_NAME = "rwillh11/mdeberta_groups_2.0"
MAX_LENGTH = 512
STRIDE     = 256

def train(data_file, output_dir):
    # load and parse the raw lists (e.g. ['Government', 'Rebels', ...])
    df = pd.read_csv(data_file)
    df["actors"] = df["regularized_actors"].apply(ast.literal_eval)

    # binarize the labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["actors"])
    df_labels = pd.DataFrame(y, columns=mlb.classes_)
    df = pd.concat([df, df_labels], axis=1)
    df["labels"] = df[mlb.classes_].values.tolist()

    # explicitly set label2id/id2label for new head
    label_list = mlb.classes_.tolist()
    id2label = {i: lbl for i, lbl in enumerate(label_list)}
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    # build HF dataset and split
    ds = Dataset.from_pandas(df[["clause","labels"]])
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    # tokenize and chunk each example into overlapping windows
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    train_tokenized = train_ds.map(
        lambda ex: chunk_and_tokenize(ex, tokenizer, MAX_LENGTH, STRIDE),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tokenized = val_ds.map(
        lambda ex: chunk_and_tokenize(ex, tokenizer, MAX_LENGTH, STRIDE),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    # format for pytorch training
    train_tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    val_tokenized.set_format(type="torch",   columns=["input_ids","attention_mask","labels"])

    # model and config setup (with the explicit id2label/label2id)
    config = AutoConfig.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )
    # load the new model
    new_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True
    )

    # show which labels exist in the new head
    new_lab2id = new_model.config.label2id
    print("New head labels:", list(new_lab2id.keys()))

    # warm start the head from the original 44-way model
    # start by loading original checkpoint
    old_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        problem_type="multi_label_classification",
    )
    old_lab2id = old_model.config.label2id

    # define the warm-start map
    warm_map = {
        "Donors":                       "Investors And Stakeholders",
        "Government":                   "Politicians",
        "Mediators":                    "Civil Servants",
        "Member States":                "Citizens",
        "Pirates":                      "Criminals",
        "Rebels":                       "Criminals",
        "Refugees":                     "Migrants And Refugees",
        "Regional Organizations":       "Ethnic And National Communities", # TALK ABOUT REGIONAL ORGANIZATIONS AND STAKEHOLDERS
        "Regional Stakeholders":        "Ethnic And National Communities",
        "Peacekeeping":                 "Military Personnel",
        "Troop Contributing Countries": "Military Personnel",
        "Terrorists":                   "Criminals",
        "Stakeholders":                 "Investors And Stakeholders",
        "Secretary General":            "Politicians",
        "Security Council":             "Politicians",
        "Permanent Members":            "Politicians",
    }

    # filter for valid mappings (shouldn't be skipping anything, but hypothetically you could)
    valid_warm = {
        new_lab: old_lab
        for new_lab, old_lab in warm_map.items()
        if new_lab in new_lab2id and old_lab in old_lab2id
    }
    skipped = set(warm_map) - set(valid_warm)
    if skipped:
        print(f"WARNING--Skipping warm-start for missing labels: {skipped}")

    # copy weights and biases for valid labels
    old_w = old_model.classifier.weight.data
    old_b = old_model.classifier.bias.data
    new_w = new_model.classifier.weight.data
    new_b = new_model.classifier.bias.data
    for new_lab, old_lab in valid_warm.items():
        ni = new_lab2id[new_lab]
        oi = old_lab2id[old_lab]
        new_w[ni].copy_(old_w[oi])
        new_b[ni].copy_(old_b[oi])

    model = new_model

    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        seed=42,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=multi_label_metrics
    )

    # train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print("Final evaluation:", eval_results)

    trainer.save_model(output_dir)
    with open(os.path.join(output_dir, "eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    # save the label classes for inference
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"classes": mlb.classes_.tolist()}, f, indent=2)


def predict(data_file, model_dir, output_file):
    # CHOOSE OPTION 1 IF YOU'RE LOADING A MODEL FROM YOUR LOCAL / CHOOSE OPTION 2 IF YOU'RE LOADING FROM HUGGINGFACE
    
    # // OPTION 1
    # load the tokenizer and label map
    #tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #with open(os.path.join(model_dir, "label_map.json")) as f:
    #    classes = json.load(f)["classes"]

    #config = AutoConfig.from_pretrained(
    #    model_dir,
    #    num_labels=len(classes),
    #    problem_type="multi_label_classification"
    #)
    #model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)

    # // OPTION 2
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 2. derive the class list directly from the model's config -----------------
    id2label = model.config.id2label          # dict: {0: "Government", 1: "Rebels", …}
    classes  = [lbl for _, lbl in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
    # // OPTIONS END

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    df = pd.read_csv(data_file)

    df = df.sample(n=1000, random_state=42)
    
    df["clause"] = df["clause"].astype(str)

    preds_list = []
    for clause in df["clause"]:
        enc = tokenizer(
            clause,
            return_overflowing_tokens=True,
            truncation=True,
            padding="max_length",       
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        mask      = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            probs  = torch.sigmoid(logits).mean(dim=0).cpu().numpy()
        preds = [classes[i] for i,p in enumerate(probs) if p >= 0.5] # change this threshold if you want to get more generous with actor prediction
        if not preds:
            preds = [classes[int(np.argmax(probs))]] # we include this so we're always predicting AT LEAST ONE actor
        preds_list.append(preds)

    df["predicted_actors"] = preds_list
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["train","predict"], required=True)
    p.add_argument("--data_file",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./model_out")
    p.add_argument("--model_dir",  type=str, help="for predict")
    p.add_argument("--output_file",type=str, default="predictions.csv")
    args = p.parse_args()

    if args.mode == "train":
        train(args.data_file, args.output_dir)
    else:
        predict(args.data_file, args.model_dir, args.output_file)