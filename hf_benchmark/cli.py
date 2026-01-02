import click
from hf_benchmark.runner import load_model, run_once
from hf_benchmark.profiler import profile_inference

@click.command()
@click.argument("model")
@click.option("--tokens", default=32, help="Number of tokens to generate")
def main(model, tokens):
    print(f"⏳ Loading {model}...")
    model_obj, tokenizer, device = load_model(model)

    task = lambda: run_once(model_obj, tokenizer, device, max_new_tokens=tokens)

    stats = profile_inference(task, token_count=tokens)

    print(f"\n✅ Device: {device.type.upper()}")
    print("-" * 30)
    print(f"📊 Latency (P50) : {stats['p50']:.2f} ms")
    print(f"🚀 Throughput    : {stats['throughput']:.2f} tokens/sec")
    print(f"💾 Peak Memory   : {stats['memory_mb']:.2f} MB") # <--- New Line
    print("-" * 30)

if __name__ == "__main__":
    main()