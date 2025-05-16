#!/usr/bin/env python3
"""
Simple Repetition Test for Fixed ContextWormhole

This script demonstrates how our fixed implementation helps reduce repetition
in generated text using a simple prompt.
"""

import torch
import numpy as np
from contextwormhole import ContextWormholeModel, ExtendedContextConfig

def analyze_text(generated_text):
    """Analyze repetition in generated text."""
    words = generated_text.split()
    unique_words = set(words)
    
    results = {
        "total_words": len(words),
        "unique_words": len(unique_words),
        "uniqueness_ratio": len(unique_words) / len(words) if len(words) > 0 else 0,
        "repeated_phrases": []
    }
    
    # Check for repeated phrases (3+ word sequences)
    if len(words) > 6:
        for i in range(len(words) - 3):
            phrase = " ".join(words[i:i+3])
            rest_of_text = " ".join(words[i+3:])
            if phrase in rest_of_text and phrase not in results["repeated_phrases"]:
                results["repeated_phrases"].append(phrase)
    
    return results

def run_test(prompt, strategy, temperature, device, model_name="distilgpt2", max_new_tokens=100):
    """Run a single test with the given parameters."""
    # Create a model with appropriate configuration
    config = ExtendedContextConfig(
        window_size=256,
        overlap=64,
        chunk_size=256,
        summary_length=64,
        sink_tokens=16,
        temperature=temperature,
        verbose=False  # Disable verbose logging
    )
    
    model = ContextWormholeModel(model_name, device=device, **config.__dict__)
    
    # Generate text with the specified strategy
    if strategy == "sliding_window":
        result = model.sliding_window_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    elif strategy == "hierarchical":
        result = model.hierarchical_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    elif strategy == "attention_sink":
        result = model.attention_sink_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    
    # Extract the generated text
    generated_text = result[len(prompt):]
    
    # Analyze repetition
    analysis = analyze_text(generated_text)
    
    return {
        "generated_text": generated_text,
        "analysis": analysis
    }

def main():
    """Run repetition tests 10 times for each strategy."""
    print("🔄 Simple Repetition Test (10 runs per strategy)")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use a small model for demonstration
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    
    # Create a simple prompt that tends to cause repetition
    prompt = "Once upon a time, there was a"
    
    print(f"\nPrompt: \"{prompt}\"")
    print("=" * 60)
    
    # Test different strategies
    strategies = [
        ("Default (Low Temperature)", "sliding_window", 0.5),
        ("Default (High Temperature)", "sliding_window", 1.0),
        ("Attention Sink", "attention_sink", 0.8),
    ]
    
    # Number of runs per strategy
    num_runs = 10
    
    # Store results for each strategy
    all_results = {}
    
    for name, strategy, temperature in strategies:
        print(f"\n{name} Strategy (Running {num_runs} times):")
        print("-" * 60)
        
        strategy_results = []
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...", end="", flush=True)
            
            # Run the test
            result = run_test(prompt, strategy, temperature, device)
            strategy_results.append(result)
            
            print(" done")
        
        # Store results
        all_results[name] = strategy_results
        
        # Calculate statistics
        uniqueness_ratios = [r["analysis"]["uniqueness_ratio"] for r in strategy_results]
        repeated_phrases_counts = [len(r["analysis"]["repeated_phrases"]) for r in strategy_results]
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"- Average uniqueness ratio: {np.mean(uniqueness_ratios):.2f} (higher is better)")
        print(f"- Average repeated phrases: {np.mean(repeated_phrases_counts):.2f} (lower is better)")
        print(f"- Best uniqueness ratio: {np.max(uniqueness_ratios):.2f}")
        print(f"- Worst uniqueness ratio: {np.min(uniqueness_ratios):.2f}")
        
        # Print best and worst examples
        best_idx = np.argmax(uniqueness_ratios)
        worst_idx = np.argmin(uniqueness_ratios)
        
        print("\nBest example:")
        print(f"\"{strategy_results[best_idx]['generated_text']}\"")
        
        print("\nWorst example:")
        print(f"\"{strategy_results[worst_idx]['generated_text']}\"")
        
        print("-" * 60)
    
    # Print comparison
    print("\n📊 Strategy Comparison:")
    print("-" * 60)
    for name in all_results:
        uniqueness_ratios = [r["analysis"]["uniqueness_ratio"] for r in all_results[name]]
        repeated_phrases_counts = [len(r["analysis"]["repeated_phrases"]) for r in all_results[name]]
        print(f"{name}:")
        print(f"- Avg uniqueness ratio: {np.mean(uniqueness_ratios):.2f}")
        print(f"- Avg repeated phrases: {np.mean(repeated_phrases_counts):.2f}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()