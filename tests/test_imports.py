#!/usr/bin/env python3
"""Test that all imports work correctly"""

def test_imports():
    """Test that all modules can be imported correctly"""
    # Base modules
    from audio_vectoring.processors.base import BaseAudioProcessor
    
    # Processors
    from audio_vectoring.processors.whisper_processor import WhisperAudioProcessor
    from audio_vectoring.processors.wav2vec_processor import Wav2VecProcessor
    from audio_vectoring.processors.clap_processor import ClapProcessor
    
    # Chunking modules
    from audio_vectoring.chunking.text_chunking import (
        chunk_text_fixed_size,
        chunk_text_recursive,
        chunk_text_by_sentences
    )
    from audio_vectoring.chunking.audio_chunking import (
        chunk_audio_by_spectrogram,
        chunk_audio_by_semantic_shift
    )
    
    # Embeddings
    from audio_vectoring.embeddings.clap_embedding import (
        ClapAudioLoader,
        ClapEmbeddingFunction
    )
    
    # Storage
    from audio_vectoring.storage.qdrant_connector import QdrantConnector
    
    # Utils
    from audio_vectoring.utils.audio_utils import (
        load_audio,
        save_audio,
        chunk_audio_fixed_size,
        chunk_audio_by_silence,
        ensure_dir
    )
    
    print("All imports successful!")

if __name__ == "__main__":
    test_imports() 