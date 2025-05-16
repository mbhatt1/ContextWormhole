#!/usr/bin/env python
# demo.py - Advanced Demo
# ======================

"""
ContextWormhole Advanced Demo

This script demonstrates advanced usage of ContextWormhole, including
custom configurations for different use cases.
"""

import torch
from contextwormhole import ContextWormholeModel, ExtendedContextConfig

def main():
    """Demonstrate advanced usage of ContextWormhole."""
    print("🌌 ContextWormhole Advanced Demo")
    print("="*50)
    
    # Check if CUDA is available for faster processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("\nContextWormhole is designed to handle long contexts by using innovative")
    print("techniques like position ID recycling, sliding windows, hierarchical")
    print("processing, and attention sink mechanisms.")
    
    # Create a conversation history for demonstration
    conversation_history = """
    User: I'm working on a research project about climate change. Can you help me understand the main factors contributing to global warming?
    
    Assistant: Of course! Global warming is primarily caused by the greenhouse effect, where certain gases in the atmosphere trap heat from the sun. The main factors contributing to this include:
    
    1. Carbon dioxide (CO2) emissions: Primarily from burning fossil fuels like coal, oil, and natural gas for electricity, transportation, and industry.
    
    2. Methane emissions: From sources like livestock farming, rice cultivation, and the decay of organic waste in landfills.
    
    3. Deforestation: Trees absorb CO2, so cutting down forests reduces the Earth's capacity to remove this greenhouse gas from the atmosphere.
    
    4. Industrial processes: Manufacturing of cement, steel, and other products releases significant greenhouse gases.
    
    5. Agricultural practices: Certain farming methods and the use of fertilizers release nitrous oxide, another potent greenhouse gas.
    
    Would you like me to elaborate on any of these factors or discuss potential solutions?
    
    User: Yes, I'd like to know more about potential solutions to address climate change. What are the most promising approaches?
    
    Assistant: There are many promising approaches to address climate change, ranging from policy changes to technological innovations and individual actions. Here are some of the most significant solutions:
    
    1. Transition to renewable energy:
       - Solar and wind power
       - Hydroelectric and geothermal energy
       - Development of better energy storage technologies
    
    2. Improve energy efficiency:
       - More efficient buildings, appliances, and industrial processes
       - Smart grids and improved electricity transmission
       - Energy-efficient transportation systems
    
    3. Sustainable transportation:
       - Electric vehicles and improved public transportation
       - Sustainable aviation fuels and more efficient aircraft
       - Urban planning that reduces the need for long commutes
    
    4. Natural climate solutions:
       - Reforestation and afforestation (planting new forests)
       - Protecting existing forests and other carbon sinks
       - Regenerative agriculture practices that sequester carbon in soil
    
    5. Policy and economic measures:
       - Carbon pricing (taxes or cap-and-trade systems)
       - Regulations on emissions
       - International agreements like the Paris Climate Accord
    
    6. Industrial innovations:
       - Carbon capture and storage technologies
       - Alternative materials with lower carbon footprints
       - Circular economy approaches that reduce waste
    
    7. Individual and community actions:
       - Dietary changes (reducing meat consumption, especially beef)
       - Reducing consumption and waste
       - Community-based renewable energy projects
    
    Which of these areas would you like to explore further for your research project?
    
    User: I'm particularly interested in carbon capture technologies. Could you explain more about how they work and their current state of development?
    """
    
    print("\n1️⃣ Basic Usage with Default Configuration")
    print("-" * 60)
    
    # Create a model with default configuration
    basic_model = ContextWormholeModel("gpt2", device=device)
    
    # Generate a response using the attention sink strategy (best for conversations)
    print("Generating response with default configuration...")
    basic_response = basic_model.attention_sink_generate(
        conversation_history, 
        max_new_tokens=100
    )
    
    # Display only the newly generated text
    basic_generated = basic_response[len(conversation_history):]
    print("\nGenerated response:")
    print("-" * 40)
    print(basic_generated)
    print("-" * 40)
    
    print("\n2️⃣ Custom Configuration for Detailed Responses")
    print("-" * 60)
    
    # Create a custom configuration for detailed, creative responses
    detailed_config = ExtendedContextConfig(
        # Increase the max training length to handle longer contexts
        max_training_length=2048,
        # Larger window size for more context
        window_size=512,
        # More overlap between windows for better coherence
        overlap=128,
        # Larger chunk size for hierarchical processing
        chunk_size=512,
        # Longer summaries for better context preservation
        summary_length=128,
        # Higher temperature for more creative responses
        temperature=0.9,
        # Higher top_p for more diverse responses
        top_p=0.95,
        # More sink tokens for better attention on key parts of the conversation
        sink_tokens=24,
        # Enable verbose output to see what's happening
        verbose=True
    )
    
    # Create a model with the detailed configuration
    detailed_model = ContextWormholeModel("gpt2", device=device, **detailed_config.__dict__)
    
    # Generate a response using the attention sink strategy
    print("Generating response with detailed configuration...")
    detailed_response = detailed_model.attention_sink_generate(
        conversation_history, 
        max_new_tokens=100
    )
    
    # Display only the newly generated text
    detailed_generated = detailed_response[len(conversation_history):]
    print("\nGenerated response:")
    print("-" * 40)
    print(detailed_generated)
    print("-" * 40)
    
    print("\n3️⃣ Configuration for Focused, Concise Responses")
    print("-" * 60)
    
    # Create a custom configuration for focused, concise responses
    focused_config = ExtendedContextConfig(
        # Standard max training length
        max_training_length=1024,
        # Smaller window size for more focus on recent context
        window_size=256,
        # Less overlap for more efficiency
        overlap=32,
        # Smaller chunk size for more focused processing
        chunk_size=256,
        # Shorter summaries for more concise context
        summary_length=64,
        # Lower temperature for more focused, deterministic responses
        temperature=0.7,
        # Lower top_p for more focused responses
        top_p=0.85,
        # Fewer sink tokens for more focus on recent context
        sink_tokens=8,
        # Enable verbose output
        verbose=True
    )
    
    # Create a model with the focused configuration
    focused_model = ContextWormholeModel("gpt2", device=device, **focused_config.__dict__)
    
    # Generate a response using the attention sink strategy
    print("Generating response with focused configuration...")
    focused_response = focused_model.attention_sink_generate(
        conversation_history, 
        max_new_tokens=100
    )
    
    # Display only the newly generated text
    focused_generated = focused_response[len(conversation_history):]
    print("\nGenerated response:")
    print("-" * 40)
    print(focused_generated)
    print("-" * 40)
    
    print("\n✅ Demo completed!")
    print("\nThis demo shows how you can customize ContextWormhole's configuration")
    print("to achieve different types of responses from the same model.")
    print("Experiment with different settings to find what works best for your use case!")
    print("\nKey configuration parameters:")
    print("- max_training_length: Maximum context length the model was trained on")
    print("- window_size: Size of sliding window for processing")
    print("- overlap: Overlap between consecutive windows")
    print("- chunk_size: Size of chunks for hierarchical processing")
    print("- summary_length: Length of summaries in hierarchical processing")
    print("- sink_tokens: Number of sink tokens for attention sink mechanism")
    print("- temperature: Controls randomness in generation (higher = more random)")
    print("- top_p/top_k: Control diversity of generated text")

if __name__ == "__main__":
    main()