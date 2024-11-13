# from transformers import AutoTokenizer, pipeline
# import torch

# rm_tokenizer = AutoTokenizer.from_pretrained("Jennny/merged_llama3_helpfulness_rm")
# # device = 0  # accelerator.device
# rm_pipe = pipeline(
#     "sentiment-analysis",
#     model="Jennny/merged_llama3_helpfulness_rm",
#     device_map="auto",
#     # device=device,
#     tokenizer=rm_tokenizer,
#     model_kwargs={"torch_dtype": torch.bfloat16},
# )

# pipe_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

# chat = [
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
# ]
# # You can prepare a list of texts like [text1, text2, ..., textn] and get rewards = [reward1, reward2, ..., rewardn]
# test_texts = [
#     rm_tokenizer.apply_chat_template(
#         chat, tokenize=False, add_generation_prompt=False
#     ).replace(rm_tokenizer.bos_token, "")
# ]
# pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
# rewards = [output[0]["score"] for output in pipe_outputs]
# print(rewards)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize the tokenizer
rm1_tokenizer = AutoTokenizer.from_pretrained("Jennny/merged_llama3_helpfulness_rm")
rm1_tokenizer.pad_token = rm1_tokenizer.eos_token

# Initialize the model for sequence classification
rm1_path = "Jennny/merged_llama3_helpfulness_rm"
RM1 = AutoModelForSequenceClassification.from_pretrained(
    rm1_path, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
)
RM1.config.pad_token_id = rm1_tokenizer.pad_token_id
RM1.eval()

# Define the chat for generating inputs
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
]

# Prepare the input text
input_text = rm1_tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=False
).replace(rm1_tokenizer.bos_token, "")

# Tokenize the input text for the model
inputs = rm1_tokenizer(
    input_text, return_tensors="pt", padding=True, truncation=True
).to(
    "cuda"
)  # Change to "cpu" if not using GPU

# Pass the inputs through the model to get the logits
with torch.no_grad():
    rm1_out = RM1(**inputs)

# Extract and process the logits to get the rewards (score)
rewards1 = rm1_out.logits.flatten().to(
    "cpu"
)  # Move to CPU if needed for further processing
print(rewards1)
