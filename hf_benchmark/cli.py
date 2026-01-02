import click
from hf_benchmark.runner import load_model, run_once
from hf_benchmark.profiler import profile_inference

@click.command()
@click.argument("model")
@click.option("--tokens", default=32, help="Number of tokens to generate")
def main(model, tokens):
    # 1. Load (returns device too!)
    model_obj, tokenizer, device = load_model(model)

    # 2. Profile
    # We pass 'device' into run_once so it knows where to send inputs
    stats = profile_inference(
        lambda: run_once(model_obj, tokenizer, device, max_new_tokens=tokens)
    )

    # 3. Report
    print(f"\n✅ Device: {device}")
    print("📊 Stats (ms):")
    for k, v in stats.items():
        print(f"   {k}: {v:.2f}")

if __name__ == "__main__":
    main()