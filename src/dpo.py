import os
import gc
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig, setup_chat_format
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import login
import wandb
import os


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size per device during training"}
    )
    gradient_accumulation_steps: int = field(
        default=16, metadata={"help": "Number of gradient accumulation steps"}
    )
    model_name: Optional[str] = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    save_every_steps: Optional[int] = field(
        default=50,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=50,
        metadata={"help": "Eval the model every x steps"},
    )
    train_dataset: Optional[str] = field(
        default="Jennny/ultrafeedback_binarized_helpfulness_prefs",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    hf_token: Optional[str] = field(
        default="",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    hub_repo_name: Optional[str] = field(
        default="llama3_helpful_dpo", metadata={"help": "Hub repository name"}
    )
    output_path: Optional[str] = field(
        default="./models/llama3_helpful_dpo",
        metadata={"help": "The dir for output model"},
    )
    wandb_project: Optional[str] = field(
        default="dpo",
        metadata={"help": "WandB project name for logging"},
    )
    wandb_name: Optional[str] = field(
        default="llama3_helpful",
        metadata={"help": "WandB run name for logging"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
wandb.init(project=script_args.wandb_project, name=script_args.wandb_name)

base_model = script_args.model_name

if torch.cuda.get_device_capability()[0] >= 8:
    # !pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16


tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"  # to prevent cutting off last generation

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)
# Reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)

# Load dataset
dataset = load_dataset(script_args.train_dataset, split="train_prefs")
dataset = dataset.shuffle(seed=42).select(range(500))


def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row


dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)

training_args = DPOConfig(
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    save_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_steps=script_args.save_every_steps,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    # max_steps=200,
    num_train_epochs=1,
    logging_steps=10,
    output_dir=script_args.output_path,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
    beta=0.1,
    max_prompt_length=1024,
    max_length=1512,
    force_use_ref_model=True,
)

dpo_trainer = DPOTrainer(
    model,
    ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dpo_config=dpo_config,
)
dpo_trainer.train()
tokenizer.push_to_hub(script_args.hub_repo_name)

# Push the model and tokenizer to the Hugging Face Hub
dpo_trainer.save_model(script_args.hub_repo_name)
dpo_trainer.push_to_hub(script_args.hub_repo_name)
