from hf_benchmark.profiler import profile_model

def run_benchmark(model_name: str, text: str):
    return profile_model(model_name, text)
