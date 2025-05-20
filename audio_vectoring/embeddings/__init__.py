"""
Embeddings module

This module contains embedding functions for different audio vectorization approaches.
"""

from audio_vectoring.embeddings.clap_embedding import (
    ClapAudioLoader,
    ClapEmbeddingFunction
)

__all__ = [
    'ClapAudioLoader',
    'ClapEmbeddingFunction'
]
