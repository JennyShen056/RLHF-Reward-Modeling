from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch

# Define the model and adapter repository names
base_model_name = (
    "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your base model name if different
)
adapter_repo_name = "Jennny/llama3_helpfulness_rm"  # Repository name where the adapter is stored on the Hub

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)

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
