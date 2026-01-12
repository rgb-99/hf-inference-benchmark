# HF Inference Benchmark

[![CI Pipeline](https://github.com/rgb-99/hf-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/rgb-99/hf-inference-benchmark/actions)
[![PyPI](https://img.shields.io/pypi/v/hf-inference-benchmark?color=orange)](https://pypi.org/project/hf-inference-benchmark/)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade CLI tool to benchmark the **Latency**, **Throughput**, and **Memory Usage** of Hugging Face models. Designed for engineers who need to know if a model will crash their production server.

## User Installation
```bash
pip install hf-inference-benchmark
```  

## Key Features
* **Latency Profiling:** Measures P50, P95, and Average inference time.
* **Throughput Metrics:** Calculates real-time generation speed (Tokens/sec).
* **Memory Tracking:** Monitors peak RAM (CPU) or VRAM (GPU) usage.
* **Device Agnostic:** Automatically detects and switches between `CUDA` (GPU) and `CPU`.
* **Production Safe:** Handles warmup runs and GPU synchronization for accurate stats.

## Installation

**From Source (Developer Mode):**
```bash
# Clone the repo
git clone [https://github.com/rgb-99/hf-inference-benchmark.git](https://github.com/rgb-99/hf-inference-benchmark.git)

# Install in editable mode
cd hf-inference-benchmark
pip install -e .
```

## Usage

Run the benchmark on any Hugging Face model. The tool handles downloading and loading automatically.

**Basic Run:**
```bash
hf-bench facebook/opt-125m
```
