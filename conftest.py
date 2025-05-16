# conftest.py - Shared Test Fixtures
# ====================================

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

@pytest.fixture
def mock_device():
    """Provide consistent device mocking."""
    with patch('torch.cuda.is_available', return_value=False):
        yield torch.device('cpu')

@pytest.fixture
def mock_config():
    """Provide a mock model config."""
    config = Mock()
    config.max_position_embeddings = 512
    config.n_positions = None
    config.n_ctx = None
    return config

@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer with consistent behavior."""
    tokenizer = Mock()
    tokenizer.encode.return_value = list(range(100))  # 100 tokens by default
    tokenizer.decode.return_value = "Generated response text"
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = None
    return tokenizer

@pytest.fixture
def mock_model(mock_config, mock_device):
    """Provide a mock model with all necessary methods."""
    model = Mock()
    model.config = mock_config
    model.device = mock_device
    
    # Mock generation output
    output = Mock()
    output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model.generate.return_value = output
    
    # Mock other methods
    model.eval.return_value = model
    model.to.return_value = model
    model.cuda.return_value = model
    model.cpu.return_value = model
    
    return model

@pytest.fixture
def complete_mock_model(mock_model, mock_tokenizer):
    """Provide a complete mock model with tokenizer attached."""
    mock_model.tokenizer = mock_tokenizer
    return mock_model

@pytest.fixture
def temp_model_dir():
    """Create temporary directory for mock model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock config.json
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, 'w') as f:
            f.write('{"max_position_embeddings": 1024, "model_type": "gpt2"}')
        
        # Create mock tokenizer files
        tokenizer_path = os.path.join(tmpdir, "tokenizer.json")
        with open(tokenizer_path, 'w') as f:
            f.write('{"version": "1.0"}')
        
        yield tmpdir

@pytest.fixture
def sample_long_text():
    """Provide sample long text for testing."""
    return "This is a sample long text. " * 200  # About 1000+ words

@pytest.fixture
def sample_short_text():
    """Provide sample short text for testing."""
    return "This is a short text for testing."

@pytest.fixture
def mock_generate_response():
    """Mock a typical generation response."""
    def _generate(input_ids, max_new_tokens=50, **kwargs):
        # Simulate adding new tokens to input
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        new_tokens = torch.randint(1, 1000, (batch_size, max_new_tokens))
        
        output = Mock()
        output.sequences = torch.cat([input_ids, new_tokens], dim=1)
        return output
    
    return _generate

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration constants."""
    return {
        'default_max_length': 512,
        'test_window_size': 256,
        'test_overlap': 50,
        'test_chunk_size': 128,
        'test_summary_length': 32,
        'test_sink_tokens': 4,
        'test_temperature': 0.7,
        'test_max_new_tokens': 100
    }

@pytest.fixture
def capture_warnings():
    """Capture warnings during tests."""
    with pytest.warns(None) as warning_list:
        yield warning_list

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import tracemalloc
    
    class MemoryTracker:
        def __init__(self):
            self.snapshots = []
        
        def start(self):
            tracemalloc.start()
            self.snapshots.append(tracemalloc.take_snapshot())
        
        def snapshot(self):
            if tracemalloc.is_tracing():
                self.snapshots.append(tracemalloc.take_snapshot())
        
        def stop(self):
            if tracemalloc.is_tracing():
                self.snapshots.append(tracemalloc.take_snapshot())
                tracemalloc.stop()
        
        def get_memory_diff(self, start_idx=0, end_idx=-1):
            if len(self.snapshots) >= 2:
                start_snapshot = self.snapshots[start_idx]
                end_snapshot = self.snapshots[end_idx]
                top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
                return sum(stat.size_diff for stat in top_stats)
            return 0
    
    return MemoryTracker()

# Parameterized test data
@pytest.fixture(params=[
    ("sliding_window", {"window_size": 256, "overlap": 50}),
    ("hierarchical", {"chunk_size": 128, "summary_length": 32}),
    ("attention_sink", {"sink_tokens": 4})
])
def strategy_params(request):
    """Provide different strategy parameters for parameterized tests."""
    return request.param

@pytest.fixture(params=[100, 500, 1000, 2000, 5000])
def input_lengths(request):
    """Provide different input lengths for testing."""
    return request.param

@pytest.fixture(params=[0.1, 0.5, 0.7, 1.0, 1.5])
def temperature_values(request):
    """Provide different temperature values for testing."""
    return request.param

# Error simulation fixtures
@pytest.fixture
def mock_error_conditions():
    """Provide various error conditions for testing."""
    return {
        'out_of_memory': RuntimeError("CUDA out of memory"),
        'model_not_found': FileNotFoundError("No such file or directory"),
        'invalid_token': ValueError("Invalid token ID"),
        'generation_error': RuntimeError("Generation failed"),
        'tokenizer_error': AttributeError("Tokenizer method not found")
    }

@pytest.fixture
def patch_transformers():
    """Patch transformers imports for testing."""
    with patch('contextwormhole.AutoTokenizer') as mock_tokenizer_class, \
         patch('contextwormhole.AutoModelForCausalLM') as mock_model_class:
        yield {
            'tokenizer_class': mock_tokenizer_class,
            'model_class': mock_model_class
        }