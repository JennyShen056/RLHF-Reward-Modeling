from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import wandb
import os


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1)
    deepspeed: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct")
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    max_length: Optional[int] = field(default=4096)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    output_dir: Optional[str] = field(default="./models/llama3_reg_rm")
    hub_model_name: Optional[str] = field(default="llama3_reg_rm")
    wandb_project: Optional[str] = field(default="llama3_reg_reward_model")
    wandb_name: Optional[str] = field(default="llama3_reg_rm")


class ScoreRewardModel(nn.Module):
    def __init__(self, base_model, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, **inputs):
        outputs = self.base_model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][:, 0, :]
        score = self.score_head(hidden_state) * 4  # Scale to 0-4 range
        return score

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()


@dataclass
class ScoreDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "labels" in features[0]:  # Changed from "score" to "labels"
            batch["labels"] = torch.tensor(
                [f["labels"] for f in features], dtype=torch.float
            )
        return batch


def build_dataset(tokenizer, split="train", eval_split=0.1):
    dataset = load_dataset("nvidia/HelpSteer2", split=split)

    def tokenize(sample):
        # Combine prompt and response using chat template
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize text
        tokenized = tokenizer(
            full_text, truncation=True, max_length=tokenizer.model_max_length
        )

        # Add helpfulness score
        tokenized["labels"] = float(sample["helpfulness"])
        return tokenized

    processed_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=8,
    )

    if split == "train" and eval_split > 0:
        # Split the processed dataset into train and eval
        train_test_dict = processed_dataset.train_test_split(
            test_size=eval_split, seed=42
        )
        return train_test_dict["train"], train_test_dict["test"]

    return processed_dataset, None  # Return the dataset and None for non-train splits


class ScoreRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        predicted_scores = model(**inputs)
        loss_fct = nn.MSELoss()
        loss = loss_fct(predicted_scores.squeeze(), labels)

        if return_outputs:
            return loss, {"predictions": predicted_scores}
        return loss


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.squeeze()
    labels = eval_pred.label_ids

    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - labels))

    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "model": args.model_name,
            "epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            "optimizer": args.optim,
            "scheduler": args.lr_scheduler_type,
            "gradient_accumulation": args.gradient_accumulation_steps,
        },
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False  # For gradient checkpointing

    # Create reward model
    model = ScoreRewardModel(base_model)

    # Prepare datasets - split train into train and eval
    train_dataset, eval_dataset = build_dataset(tokenizer, "train", eval_split=0.1)
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="wandb",  # Enable wandb logging
        remove_unused_columns=False,  # Added this line
    )

    # Initialize trainer
    trainer = ScoreRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ScoreDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=args.max_length
        ),
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()
    tokenizer.push_to_hub(args.hub_repo_name)

    # Save model
    trainer.save_model(args.hub_repo_name)
    trainer.push_to_hub(args.hub_repo_name)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
