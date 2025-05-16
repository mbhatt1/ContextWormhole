#!/usr/bin/env python
# benchmark_contextwormhole.py - Performance Benchmarks
# ===================================================

"""
ContextWormhole Performance Benchmarks

This script benchmarks the performance of different context extension strategies
in ContextWormhole, measuring memory usage and processing time.
"""

import time
import gc
import os
import psutil
import torch
from contextwormhole import ContextWormholeModel, ExtendedContextConfig

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_strategy(model, strategy_name, input_text, max_new_tokens=50):
    """Benchmark a specific strategy.
    
    Args:
        model: ContextWormholeModel instance
        strategy_name: Name of the strategy to benchmark
        input_text: Input text to process
        max_new_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with benchmark results
    """
    # Clear cache and collect garbage
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Get initial memory usage
    start_memory = get_memory_usage()
    
    # Start timer
    start_time = time.time()
    
    # Process with the specified strategy
    if strategy_name == "sliding_window":
        result = model.sliding_window_generate(input_text, max_new_tokens=max_new_tokens)
    elif strategy_name == "hierarchical":
        result = model.hierarchical_generate(input_text, max_new_tokens=max_new_tokens)
    elif strategy_name == "attention_sink":
        result = model.attention_sink_generate(input_text, max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # End timer
    end_time = time.time()
    
    # Get final memory usage
    end_memory = get_memory_usage()
    
    # Calculate metrics
    processing_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    return {
        "strategy": strategy_name,
        "input_length": len(input_text),
        "processing_time": processing_time,
        "memory_used": memory_used,
        "output_length": len(result)
    }

def run_benchmarks(model_name="gpt2", device=None):
    """Run benchmarks for all strategies with different input lengths.
    
    Args:
        model_name: Name of the model to use
        device: Device to run on (None for auto-detection)
    """
    print(f"🔍 Benchmarking ContextWormhole with model: {model_name}")
    print(f"Device: {device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 80)
    
    # Create model with default configuration
    model = ContextWormholeModel(model_name, device=device)
    
    # Create input texts of different lengths
    input_lengths = [1000, 5000, 10000, 20000]
    input_texts = {
        length: "This is a test document for benchmarking. " * (length // 40)
        for length in input_lengths
    }
    
    # Strategies to benchmark
    strategies = ["sliding_window", "hierarchical", "attention_sink"]
    
    # Run benchmarks
    results = []
    
    for length, text in input_texts.items():
        print(f"\nBenchmarking with input length: {length} characters")
        
        for strategy in strategies:
            print(f"  - Running {strategy}...", end="", flush=True)
            result = benchmark_strategy(model, strategy, text)
            results.append(result)
            print(f" done in {result['processing_time']:.2f}s (Memory: {result['memory_used']:.2f} MB)")
    
    # Print results table
    print("\n\n📊 Benchmark Results")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Input Length':<15} {'Processing Time':<20} {'Memory Used':<15} {'Output Length':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['strategy']:<20} {result['input_length']:<15} {result['processing_time']:.2f}s{'':<13} {result['memory_used']:.2f} MB{'':<6} {result['output_length']:<15}")
    
    # Print summary
    print("\n📈 Summary")
    print("=" * 80)
    
    for strategy in strategies:
        strategy_results = [r for r in results if r['strategy'] == strategy]
        avg_time = sum(r['processing_time'] for r in strategy_results) / len(strategy_results)
        avg_memory = sum(r['memory_used'] for r in strategy_results) / len(strategy_results)
        
        print(f"{strategy}: Avg Time = {avg_time:.2f}s, Avg Memory = {avg_memory:.2f} MB")
    
    # Memory usage by input length
    print("\n📉 Memory Usage by Input Length")
    print("=" * 80)
    
    for length in input_lengths:
        length_results = [r for r in results if r['input_length'] == length]
        for strategy in strategies:
            strategy_result = next((r for r in length_results if r['strategy'] == strategy), None)
            if strategy_result:
                print(f"Input Length {length}: {strategy} = {strategy_result['memory_used']:.2f} MB")

if __name__ == "__main__":
    # Run benchmarks with default model
    run_benchmarks()