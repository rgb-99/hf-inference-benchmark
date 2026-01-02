import argparse
from hf_benchmark.runner import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Inference Benchmark Tool")
    parser.add_argument("--model", type=str, required=True, help="Name of the model (e.g. gpt2)")
    parser.add_argument("--text", type=str, default="Hello world", help="Input text to process")
    
    args = parser.parse_args()
    
    try:
        run_benchmark(args.model, args.text)
    except Exception as e:
        print(f"❌ Error: {e}")