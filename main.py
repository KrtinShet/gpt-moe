import torch
import tiktoken
from src.wednesday_mark2 import Wednesday, WednesdayConfig

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'

tokenizer = tiktoken.get_encoding("gpt2")
config = WednesdayConfig(d_model=1024, n_layer=32, top_k=4, max_seq_len=2048000, n_experts=8, n_heads=16)
model = Wednesday(config)
# model = model.to("mps")
model.save_pretrained('./data')

print("Model size:", model.count_parameters())

# text = "Hello, World!, I'm a language model "

# tokens = tokenizer.encode(text)

# print("Wednesday model initialized successfully.")

# torch.manual_seed(42)
# torch.mps.manual_seed(42)

# x = (torch.tensor(tokens, dtype=torch.long, device=device)[None, ...])
# generated = model.generate(x, max_length=20)

# print("Decoded text:", tokenizer.decode(generated[0].tolist()))
# print("Text generation completed successfully.")

# 1,309,903,872
# 2,568,342,528 2.5B
