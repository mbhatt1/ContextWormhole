#!/usr/bin/env python
# example.py - Simple Example
# ==========================

"""
ContextWormhole Simple Example

This script demonstrates how to use ContextWormhole with a real model
to handle long context inputs.
"""

from contextwormhole import ContextWormholeModel

def main():
    """Demonstrate the use of ContextWormhole with a real model."""
    print("🌌 ContextWormhole Simple Example")
    print("="*50)
    
    # Initialize the model with a real model path and proper configuration
    # Note: This will download the model if it's not already cached
    print("Loading model...")
    
    # Import the configuration class
    from contextwormhole import ExtendedContextConfig
    
    # Create a configuration optimized for long contexts
    config = ExtendedContextConfig(
        max_training_length=2048,  # Increase the max training length
        window_size=512,           # Larger window size
        overlap=128,               # Significant overlap for better coherence
        chunk_size=512,            # Larger chunk size for hierarchical processing
        summary_length=128,        # Longer summaries
        sink_tokens=32,            # More sink tokens for attention sink
        temperature=0.8,           # Standard temperature
        verbose=True               # Show verbose output to see what's happening
    )
    
    # Initialize the model with our configuration
    model = ContextWormholeModel("gpt2", **config.__dict__)
    print("Model loaded successfully with extended context configuration!")
    
    # Create a long document that exceeds the normal context length
    long_document = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed
    to natural intelligence displayed by animals including humans. AI research has been
    defined as the field of study of intelligent agents, which refers to any system that
    perceives its environment and takes actions that maximize its chance of achieving
    its goals.
    
    The term "artificial intelligence" was coined in 1956, but AI has become more popular
    today thanks to increased data volumes, advanced algorithms, and improvements in
    computing power and storage.
    
    Early AI research in the 1950s explored topics like problem solving and symbolic
    methods. In the 1960s, the US Department of Defense took interest in this type of
    work and began training computers to mimic basic human reasoning.
    
    For example, the Defense Advanced Research Projects Agency (DARPA) completed street
    mapping projects in the 1970s. And DARPA produced intelligent personal assistants
    in 2003, long before Siri, Alexa or Cortana were household names.
    
    This early work paved the way for the automation and formal reasoning that we see
    in computers today, including decision support systems and smart search systems that
    can be designed to complement and augment human abilities.
    
    While Hollywood movies and science fiction novels depict AI as human-like robots that
    take over the world, the current evolution of AI technologies isn't that scary – or
    quite that smart. Instead, AI has evolved to provide many specific benefits in every
    industry.
    """ + "AI continues to evolve and find new applications in various fields. " * 30
    
    print(f"\nDocument length: ~{len(long_document.split())} tokens")
    
    # Use different strategies for generating text
    print("\n📝 Sliding Window Strategy")
    print("-" * 50)
    print("Processing document with sliding window strategy...")
    result1 = model.sliding_window_generate(long_document, max_new_tokens=50)
    
    # Extract and display only the newly generated text
    new_text1 = result1[len(long_document):]
    print("\nGenerated text:")
    print("-" * 40)
    print(new_text1 if new_text1.strip() else "(No new content generated)")
    print("-" * 40)
    
    print("\n📝 Hierarchical Strategy")
    print("-" * 50)
    print("Processing document with hierarchical strategy...")
    result2 = model.hierarchical_generate(long_document, max_new_tokens=50)
    
    # Extract and display only the newly generated text
    new_text2 = result2[len(long_document):]
    print("\nGenerated text:")
    print("-" * 40)
    print(new_text2 if new_text2.strip() else "(No new content generated)")
    print("-" * 40)
    
    print("\n📝 Attention Sink Strategy")
    print("-" * 50)
    print("Processing document with attention sink strategy...")
    result3 = model.attention_sink_generate(long_document, max_new_tokens=50)
    
    # Extract and display only the newly generated text
    new_text3 = result3[len(long_document):]
    print("\nGenerated text:")
    print("-" * 40)
    print(new_text3 if new_text3.strip() else "(No new content generated)")
    print("-" * 40)
    
    print("\n✅ Example completed!")
    print("\nThis example demonstrates how ContextWormhole can handle long context")
    print("inputs using different strategies. Each strategy has its own strengths:")
    print("- Sliding Window: Best for documents and articles")
    print("- Hierarchical: Best for research papers with sections")
    print("- Attention Sink: Best for conversations")
    print("\nThe library automatically handles context length limitations by using")
    print("innovative techniques like position ID recycling, sliding windows,")
    print("hierarchical processing, and attention sink mechanisms.")

if __name__ == "__main__":
    main()