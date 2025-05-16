# example_usage.py - Example Usage Script
# ========================================

"""
ContextWormhole Library - Example Usage

This script demonstrates how to use the ContextWormhole library
with different strategies for handling long context.
"""

from contextwormhole import (
    ContextWormholeModel,
    ExtendedContextConfig,
    sliding_window,
    hierarchical_context,
    attention_sink,
    extended_context,
    configure_extended_context,
    create_extended_model
)

# For demo purposes, we'll create mock scenarios
# In real usage, you would have actual model paths

def demo_basic_usage():
    """Demonstrate basic usage of ContextWormhole."""
    print("🌌 ContextWormhole Basic Usage Demo")
    print("="*50)
    
    # Note: In real usage, you'd use actual model paths like:
    # model = ContextWormholeModel("microsoft/DialoGPT-medium")
    
    print("Creating ContextWormhole model...")
    print("model = ContextWormholeModel('microsoft/DialoGPT-medium')")
    
    # Simulate long document
    long_document = """
    This is a very long research paper about artificial intelligence
    that exceeds the normal context length of most models...
    """ + "Content continues for many paragraphs... " * 50
    
    print(f"\nDocument length: ~{len(long_document)} characters")
    
    # Demo different strategies
    strategies = [
        ("sliding_window", "Best for documents and articles"),
        ("hierarchical", "Best for research papers with sections"),
        ("attention_sink", "Best for conversations"),
    ]
    
    for strategy, description in strategies:
        print(f"\n📝 {strategy.replace('_', ' ').title()} Strategy")
        print(f"   {description}")
        print(f"   result = model.{strategy}_generate(document, max_new_tokens=100)")
        print(f"   # Would process {len(long_document)} characters...")
    
    print("\n✅ Demo completed successfully!")

def demo_decorator_usage():
    """Demonstrate using decorators for custom functions."""
    print("\n🔧 ContextWormhole Decorator Demo")
    print("="*50)
    
    print("Using decorators on custom functions...")
    
    # Example with sliding window
    print("\n1. Sliding Window Decorator:")
    print("""
    @sliding_window(window_size=512, overlap=50)
    def process_document(model, text, **kwargs):
        return model.generate(text, **kwargs)
    
    result = process_document(model, long_text)
    """)
    
    # Example with hierarchical context
    print("\n2. Hierarchical Context Decorator:")
    print("""
    @hierarchical_context(chunk_size=256, summary_length=50)
    def analyze_paper(model, paper, **kwargs):
        return model.generate(f"Analyze: {paper}", **kwargs)
    
    analysis = analyze_paper(model, research_paper)
    """)
    
    # Example with attention sink
    print("\n3. Attention Sink Decorator:")
    print("""
    @attention_sink(sink_tokens=4)
    def continue_conversation(model, chat_history, **kwargs):
        return model.generate(chat_history, **kwargs)
    
    response = continue_conversation(model, long_conversation)
    """)

def demo_class_configuration():
    """Demonstrate class-level configuration."""
    print("\n⚙️ ContextWormhole Class Configuration Demo")
    print("="*50)
    
    print("Creating custom class with ContextWormhole integration...")
    print("""
    @configure_extended_context(
        max_training_length=1024,
        temperature=0.8,
        verbose=True
    )
    class SmartChatBot:
        def __init__(self, model_path):
            self.model = create_extended_model(model_path)
            self.tokenizer = self.model.tokenizer
        
        @sliding_window(window_size=512)
        def chat(self, message, **kwargs):
            return self.model.generate(message, **kwargs)
        
        @hierarchical_context(chunk_size=200)
        def analyze_document(self, document, **kwargs):
            return self.model.generate(f"Analyze: {document}", **kwargs)
    
    # Usage:
    bot = SmartChatBot("microsoft/DialoGPT-medium")
    response = bot.chat("Hello, how are you?")
    """)

def demo_advanced_features():
    """Demonstrate advanced features and configurations."""
    print("\n🚀 ContextWormhole Advanced Features Demo")
    print("="*50)
    
    print("1. Custom Configuration:")
    print("""
    config = ExtendedContextConfig(
        max_training_length=2048,
        window_size=1024,
        overlap=100,
        chunk_size=512,
        summary_length=64,
        sink_tokens=8,
        temperature=0.9,
        top_p=0.95,
        use_cache=True,
        verbose=True
    )
    
    model = ContextWormholeModel("gpt2-large", **config.__dict__)
    """)
    
    print("\n2. Meta-Decorator:")
    print("""
    @extended_context(strategy="sliding_window", window_size=1024)
    def smart_generate(model, prompt, **kwargs):
        return model.generate(prompt, **kwargs)
    
    # Automatically chooses the best approach
    """)
    
    print("\n3. Auto-Detection:")
    print("""
    @auto_detect_context_length
    def setup_model(model):
        # Automatically detects model's context length
        # and configures accordingly
        return model
    """)

def demo_real_world_scenarios():
    """Demonstrate real-world use cases."""
    print("\n🌍 ContextWormhole Real-World Scenarios")
    print("="*50)
    
    scenarios = [
        {
            "name": "Document Q&A",
            "description": "Analyze entire research papers or legal documents",
            "strategy": "hierarchical_context",
            "example": "model.hierarchical_generate(f'{paper}\\n\\nQ: What are the main findings?')"
        },
        {
            "name": "Long Conversations",
            "description": "Maintain context across lengthy chat sessions",
            "strategy": "attention_sink",
            "example": "model.attention_sink_generate(conversation_history)"
        },
        {
            "name": "Code Analysis", 
            "description": "Analyze large codebases or long source files",
            "strategy": "sliding_window",
            "example": "model.sliding_window_generate(f'Review this code:\\n{source_code}')"
        },
        {
            "name": "Story Writing",
            "description": "Continue long-form creative writing",
            "strategy": "sliding_window",
            "example": "model.sliding_window_generate(f'{story_so_far}\\n\\nContinue:')"
        },
        {
            "name": "Email/Report Generation",
            "description": "Generate content based on long context",
            "strategy": "hierarchical_context", 
            "example": "model.hierarchical_generate(f'Write report based on: {data}')"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Best Strategy: {scenario['strategy']}")
        print(f"   Example: {scenario['example']}")

def demo_performance_tips():
    """Demonstrate performance optimization tips."""
    print("\n⚡ ContextWormhole Performance Tips")
    print("="*50)
    
    tips = [
        {
            "tip": "Choose the Right Strategy",
            "details": [
                "sliding_window: Best for documents, articles, code",
                "hierarchical_context: Best for research papers, reports",
                "attention_sink: Best for conversations, chat histories"
            ]
        },
        {
            "tip": "Optimize Parameters",
            "details": [
                "Larger overlap = better continuity, slower processing",
                "Smaller chunk_size = more granular summaries",
                "Adjust window_size based on GPU memory"
            ]
        },
        {
            "tip": "Memory Management",
            "details": [
                "Use smaller windows for large documents",
                "Enable use_cache=True for better performance",
                "Consider device placement (CPU vs GPU)"
            ]
        },
        {
            "tip": "Batch Processing",
            "details": [
                "Process multiple documents separately",
                "Reuse model instances when possible",
                "Monitor memory usage during long sessions"
            ]
        }
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"\n{i}. {tip['tip']}:")
        for detail in tip['details']:
            print(f"   • {detail}")

def main():
    """Run all demos."""
    print("🌌 ContextWormhole Library - Complete Demo")
    print("="*60)
    print("This demo shows how to use ContextWormhole for extending")
    print("context length in transformer models beyond their limits.")
    print("="*60)
    
    # Run all demos
    demo_basic_usage()
    demo_decorator_usage()
    demo_class_configuration()
    demo_advanced_features()
    demo_real_world_scenarios()
    demo_performance_tips()
    
    print("\n🎯 Quick Start Guide:")
    print("="*30)
    print("1. Install: pip install contextwormhole")
    print("2. Import: from contextwormhole import ContextWormholeModel")
    print("3. Create model: model = ContextWormholeModel('gpt2')")
    print("4. Generate: result = model.sliding_window_generate(long_text)")
    print("\n📚 For more examples, see the GitHub repository!")
    print("🌟 Happy context warping! 🌟")

if __name__ == "__main__":
    main()