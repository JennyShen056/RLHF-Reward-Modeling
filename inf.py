from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize the tokenizer
rm1_tokenizer = AutoTokenizer.from_pretrained("Jennny/llama3_helpful_rm_full")
rm1_tokenizer.pad_token = rm1_tokenizer.eos_token

# Initialize the model for sequence classification
rm1_path = "Jennny/llama3_helpful_rm_full"
RM1 = AutoModelForSequenceClassification.from_pretrained(
    rm1_path, num_labels=1, torch_dtype=torch.float16, device_map="auto"
)
RM1.config.pad_token_id = rm1_tokenizer.pad_token_id
RM1.eval()

# Define the chats for generating inputs
chats = [
    [
        {"content": "What is the advantage of separable filters?", "role": "user"},
        {
            "content": "Separable filters are advantageous in several ways, particularly in image processing and computer graphics. They allow for efficient implementation and memory storage, enabling faster processing and improved image quality. Some advantages of separable filters include:\n\n1. Efficient Implementation: Separable filters can be applied row-by-row or column-by-column, significantly reducing the number of operations required compared to non-separable filters. This parallel processing technique leads to faster execution and improved performance.\n2. Memory Storage: Separable filters require less memory to store the filter coefficients since they can be separated into two one-dimensional filters instead of a single two-dimensional filter. This reduction in memory usage is beneficial in applications with limited memory resources.\n3. Flexibility: Separable filters offer greater flexibility in modifying and combining filtering operations. For example, you can apply different filters horizontally and vertically to achieve customized effects.\n4. Robustness: In some cases, separable filters are more robust and less sensitive to noise compared to non-separable filters. This is particularly true for certain types of edge detection and noise reduction applications.\n5. Reduced Complexity: Separable filters generally have lower computational complexity than non-separable filters. This makes them more suitable for real-time applications and systems with limited processing power.\n6. Applications: Separable filters are widely used in image and video processing, computer graphics, and many other fields like signal and speech processing, data compression, and medical imaging.\n\nOverall, separable filters provide a more efficient and effective approach to filtering operations, making them a popular choice in various applications that require fast and high-quality processing.",
            "role": "assistant",
        },
    ],
    [
        {"content": "What is the advantage of separable filters?", "role": "user"},
        {
            "content": "Separable filters are advantageous because they can be analytically factored. This makes them computationally efficient when compared to non-separable filters which require a time consuming numerical solution to calculate. Additionally, separable filters are biologically interpretable which can be useful for understanding the effects of a filtering process.",
            "role": "assistant",
        },
    ],
]

# Prepare inputs for each chat and get rewards
rewards = []
for chat in chats:
    # Prepare the input text for each chat
    input_text = rm1_tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    ).replace(rm1_tokenizer.bos_token, "")

    # Tokenize the input text
    inputs = rm1_tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True
    ).to(
        "cuda"
    )  # Change to "cpu" if not using GPU

    # Pass the inputs through the model to get logits
    with torch.no_grad():
        rm1_out = RM1(**inputs)

    # Extract and process the logits to get the rewards (score)
    reward = rm1_out.logits.flatten().item()  # Get score as a single float value
    rewards.append(reward)

print(rewards)
