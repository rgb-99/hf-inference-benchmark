import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def profile_model(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")

    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    start = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)
    end = time.time()

    print(f"Latency: {end - start:.2f}s")
