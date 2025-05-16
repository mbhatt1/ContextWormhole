#!/usr/bin/env python
# contextwormhole.py - Main Module
# ===============================

"""
ContextWormhole: A library for extending context length in transformer models.

This module provides a simple interface to the ContextWormhole library,
allowing for easy use of different context handling strategies.
"""

from contextwormhole import ContextWormholeModel

def main():
    """Demonstrate the use of ContextWormhole with different strategies."""
    print("🌌 ContextWormhole Demo")
    print("="*50)
    
    # Initialize the model with proper configuration for handling long contexts
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
    print("Model initialized with extended context configuration")
    
    # Create sample documents for different strategies
    
    # Long document for sliding window strategy
    long_document = """
    This is a very long document about artificial intelligence that exceeds 
    the normal context length of most models. It contains multiple paragraphs
    discussing various aspects of AI, its history, current applications, and
    future prospects.
    
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed
    to natural intelligence displayed by animals including humans. AI research has been
    defined as the field of study of intelligent agents, which refers to any system that
    perceives its environment and takes actions that maximize its chance of achieving
    its goals.
    
    The term "artificial intelligence" was coined in 1956, but AI has become more popular
    today thanks to increased data volumes, advanced algorithms, and improvements in
    computing power and storage.
    """ + "AI continues to evolve and find new applications in various fields. " * 50
    
    # Research paper for hierarchical strategy
    research_paper = """
    # Abstract
    
    This research paper presents a novel approach to natural language processing
    using transformer models with extended context capabilities. We demonstrate
    significant improvements in performance on long-context tasks.
    
    # Introduction
    
    Natural language processing has seen remarkable advances in recent years,
    primarily due to the development of transformer-based models. However, these
    models often struggle with long contexts due to the quadratic complexity of
    self-attention mechanisms.
    
    # Related Work
    
    Several approaches have been proposed to address the context length limitations
    of transformer models. These include sparse attention patterns, recurrent memory
    mechanisms, and hierarchical processing strategies.
    
    # Methodology
    
    Our approach combines several techniques to extend the effective context length
    of transformer models. We implement a hierarchical attention mechanism that
    processes the input at multiple levels of granularity.
    
    # Experiments
    
    We conducted extensive experiments on various long-context tasks, including
    document summarization, question answering, and long-form generation.
    
    # Results
    
    Our results demonstrate that the proposed approach significantly outperforms
    baseline models on all evaluated tasks, with particularly strong performance
    on tasks requiring understanding of long documents.
    
    # Conclusion
    
    We have presented a novel approach to extending the context length of transformer
    models, enabling them to process and understand much longer documents than
    previously possible.
    """ + "The implications of this work are far-reaching and open up new possibilities for AI applications. " * 30
    
    # Conversation history for attention sink strategy
    conversation_history = """
    User: Hello, I'm interested in learning more about artificial intelligence. Can you help me?
    
    Assistant: Of course! I'd be happy to help you learn about artificial intelligence. What specific aspects are you interested in?
    
    User: I'm particularly interested in how AI is used in healthcare. Can you tell me about that?
    
    Assistant: Certainly! AI has numerous applications in healthcare. Here are some key areas:
    
    1. Medical imaging analysis: AI can help detect abnormalities in X-rays, MRIs, and CT scans, often with accuracy comparable to or exceeding that of human radiologists.
    
    2. Drug discovery: AI accelerates the process of identifying potential drug candidates by analyzing biological data and predicting how different compounds might interact with specific targets.
    
    3. Personalized medicine: AI algorithms can analyze a patient's genetic information, medical history, and other data to help tailor treatments to individual patients.
    
    4. Clinical decision support: AI systems can assist healthcare providers by offering evidence-based recommendations for diagnosis and treatment.
    
    5. Remote patient monitoring: AI can analyze data from wearable devices to monitor patients' health status and alert healthcare providers to potential issues.
    
    Would you like me to elaborate on any of these applications?
    
    User: Yes, I'd like to know more about AI in medical imaging. How exactly does it work?
    
    Assistant: AI in medical imaging works through a process called deep learning, particularly using Convolutional Neural Networks (CNNs). Here's how it works:
    
    1. Training: The AI system is trained on thousands of medical images that have been labeled by expert radiologists. For example, it learns to recognize the visual patterns associated with conditions like tumors, fractures, or pneumonia.
    
    2. Feature extraction: The CNN automatically learns to identify relevant features in the images, from simple edges and textures to complex anatomical structures.
    
    3. Classification/Detection: Once trained, the AI can analyze new images to:
       - Classify images (e.g., "pneumonia" vs. "no pneumonia")
       - Detect and locate abnormalities (e.g., highlighting potential tumors)
       - Segment images to identify specific organs or tissues
    
    4. Quantification: AI can measure sizes, volumes, and changes over time more precisely than manual methods.
    
    5. Workflow integration: In clinical settings, AI is typically integrated into the radiologist's workflow, flagging urgent cases for immediate review or providing a "second opinion."
    
    The benefits include:
    - Reduced workload for radiologists
    - Faster diagnosis, especially in emergency situations
    - Detection of subtle abnormalities that might be missed by human eyes
    - Consistent analysis, without fatigue or distraction
    
    However, AI typically works alongside radiologists rather than replacing them, as the final interpretation still benefits from human expertise and context.
    
    User: That's fascinating! Are there any ethical concerns with using AI in healthcare?
    """ + "I'm curious about the potential risks and how they're being addressed. " * 10
    
    # Different strategies for different needs
    print("\n📝 Sliding Window Strategy (for documents)")
    print("Processing document with approximately", len(long_document.split()), "tokens...")
    result1 = model.sliding_window_generate(long_document, max_new_tokens=100)
    
    # Extract and display only the newly generated text
    new_text1 = result1[len(long_document):]
    print("\nGenerated text:")
    print("-" * 40)
    print(new_text1)
    print("-" * 40)
    
    print("\n📝 Hierarchical Strategy (for research papers)")
    print("Processing research paper with approximately", len(research_paper.split()), "tokens...")
    result2 = model.hierarchical_generate(research_paper, max_new_tokens=100)
    
    # Extract and display only the newly generated text
    new_text2 = result2[len(research_paper):]
    print("\nGenerated text:")
    print("-" * 40)
    print(new_text2)
    print("-" * 40)
    
    print("\n📝 Attention Sink Strategy (for conversations)")
    print("Processing conversation with approximately", len(conversation_history.split()), "tokens...")
    result3 = model.attention_sink_generate(conversation_history, max_new_tokens=100)
    
    # Extract and display only the newly generated text
    new_text3 = result3[len(conversation_history):]
    print("\nGenerated text:")
    print("-" * 40)
    print(new_text3)
    print("-" * 40)
    
    print("\n✅ Demo completed!")
    print("\nThis demo shows how ContextWormhole can handle long contexts using different strategies.")
    print("Each strategy has its own strengths:")
    print("- Sliding Window: Best for documents and articles")
    print("- Hierarchical: Best for research papers with sections")
    print("- Attention Sink: Best for conversations")

if __name__ == "__main__":
    main()