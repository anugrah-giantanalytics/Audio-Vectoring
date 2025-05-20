"""
Audio and text chunking module

This module contains functions for chunking audio and text data:
- Text chunking: Fixed-size, recursive, sentence-based
- Audio chunking: Fixed-size, silence-based, spectrogram-based, semantic shift
"""

from audio_vectoring.chunking.text_chunking import (
    chunk_text_fixed_size,
    chunk_text_recursive,
    chunk_text_by_sentences
)

from audio_vectoring.chunking.audio_chunking import (
    chunk_audio_by_spectrogram,
    chunk_audio_by_semantic_shift
)

__all__ = [
    'chunk_text_fixed_size',
    'chunk_text_recursive',
    'chunk_text_by_sentences',
    'chunk_audio_by_spectrogram',
    'chunk_audio_by_semantic_shift'
]
