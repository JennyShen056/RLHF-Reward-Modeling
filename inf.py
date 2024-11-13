from transformers import AutoTokenizer, pipeline
import torch

rm_tokenizer = AutoTokenizer.from_pretrained("Jennny/merged_llama3_helpfulness_rm")
device = 0  # accelerator.device
rm_pipe = pipeline(
    "sentiment-analysis",
    model="Jennny/merged_llama3_helpfulness_rm",
    # device="auto",
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

pipe_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
]
# You can prepare a list of texts like [text1, text2, ..., textn] and get rewards = [reward1, reward2, ..., rewardn]
test_texts = [
    rm_tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    ).replace(rm_tokenizer.bos_token, "")
]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
rewards = [output[0]["score"] for output in pipe_outputs]
