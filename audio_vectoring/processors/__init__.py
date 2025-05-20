"""
Audio processors module

This module contains the implementations of different audio processing approaches:
- WhisperAudioProcessor: Text-based approach using transcription
- Wav2VecProcessor: Direct audio approach
- ClapProcessor: Multimodal approach
"""

from audio_vectoring.processors.base import BaseAudioProcessor
from audio_vectoring.processors.whisper_processor import WhisperAudioProcessor
from audio_vectoring.processors.wav2vec_processor import Wav2VecProcessor
from audio_vectoring.processors.clap_processor import ClapProcessor

__all__ = [
    'BaseAudioProcessor',
    'WhisperAudioProcessor',
    'Wav2VecProcessor',
    'ClapProcessor'
]
