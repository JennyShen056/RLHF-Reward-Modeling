from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch
from huggingface_hub import login

token = "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"
login(token=token)

# Define the model and adapter repository names
base_model_name = (
    "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your base model name if different
)
adapter_repo_name = "Jennny/llama3_helpfulness_rm"  # Repository name where the adapter is stored on the Hub

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=1,
    # torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", use_fast=False
)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
print(tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = 4096
tokenizer.padding_side = "right"

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.resize_token_embeddings(len(tokenizer))

# Load the adapter and apply it to the base model
peft_model = PeftModel.from_pretrained(base_model, adapter_repo_name)

# Merge the adapter weights into the base model
merged_model = peft_model.merge_and_unload()

# Load the tokenizer (optional if needed)
tokenizer = AutoTokenizer.from_pretrained(adapter_repo_name, use_fast=False)

# Optionally, push the merged model back to the Hugging Face Hub if required
merged_model.push_to_hub("merged_llama3_helpfulness_rm")
tokenizer.push_to_hub("merged_llama3_helpfulness_rm")

print("Adapter merged with base model and saved successfully.")
