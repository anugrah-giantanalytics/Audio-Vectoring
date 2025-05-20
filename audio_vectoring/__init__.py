"""
Audio Vectorization Framework

A modular framework for comparing different audio vectorization approaches:
- Whisper (text-based)
- Wav2Vec (direct audio)
- CLAP (multimodal)
"""

__version__ = "0.1.0"

# Import processors
from audio_vectoring.processors import (
    BaseAudioProcessor,
    WhisperAudioProcessor,
    Wav2VecProcessor,
    ClapProcessor
)

# Import chunking utilities
from audio_vectoring.chunking import (
    chunk_text_fixed_size,
    chunk_text_recursive,
    chunk_text_by_sentences,
    chunk_audio_by_spectrogram,
    chunk_audio_by_semantic_shift
)

# Import embedding functions
from audio_vectoring.embeddings import (
    ClapAudioLoader,
    ClapEmbeddingFunction
)

# Import storage utilities
from audio_vectoring.storage import QdrantConnector

# Import audio utilities
from audio_vectoring.utils import (
    load_audio,
    save_audio,
    chunk_audio_fixed_size,
    chunk_audio_by_silence,
    ensure_dir
)

__all__ = [
    # Processors
    'BaseAudioProcessor',
    'WhisperAudioProcessor',
    'Wav2VecProcessor',
    'ClapProcessor',
    
    # Chunking
    'chunk_text_fixed_size',
    'chunk_text_recursive',
    'chunk_text_by_sentences',
    'chunk_audio_by_spectrogram',
    'chunk_audio_by_semantic_shift',
    
    # Embeddings
    'ClapAudioLoader',
    'ClapEmbeddingFunction',
    
    # Storage
    'QdrantConnector',
    
    # Utils
    'load_audio',
    'save_audio',
    'chunk_audio_fixed_size',
    'chunk_audio_by_silence',
    'ensure_dir'
]
