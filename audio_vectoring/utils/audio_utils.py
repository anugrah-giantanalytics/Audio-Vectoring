import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Dict, Optional, Any
from pydub import AudioSegment
from pydub.silence import split_on_silence

def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy array with specified sample rate
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio data as numpy array, sample rate)
    """
    print(f"Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def save_audio(audio: np.ndarray, file_path: str, sr: int) -> str:
    """
    Save audio data to a file
    
    Args:
        audio: Audio data as numpy array
        file_path: Output file path
        sr: Sample rate
        
    Returns:
        Path to the saved file
    """
    sf.write(file_path, audio, sr)
    return file_path

def chunk_audio_fixed_size(audio: np.ndarray, sr: int, chunk_duration_sec: int = 30) -> List[Dict[str, Any]]:
    """
    Chunk audio into fixed-size segments
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        chunk_duration_sec: Duration of each chunk in seconds
        
    Returns:
        List of dictionaries containing audio chunks and metadata
    """
    print(f"Chunking audio into fixed {chunk_duration_sec}-second segments...")
    chunk_size = sr * chunk_duration_sec
    chunks = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        # Only keep chunks that are at least half the desired size
        if len(chunk) > 0.5 * chunk_size:
            chunk_data = {
                "audio": chunk,
                "start_time": i / sr,
                "end_time": min((i + chunk_size), len(audio)) / sr,
                "chunk_id": len(chunks)
            }
            chunks.append(chunk_data)
    
    print(f"Created {len(chunks)} fixed-size audio chunks")
    return chunks

def chunk_audio_by_silence(file_path: str, min_silence_len: int = 500, 
                          silence_thresh: int = -40) -> List[Dict[str, Any]]:
    """
    Chunk audio by detecting silence using pydub
    
    Args:
        file_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        
    Returns:
        List of dictionaries containing audio chunks and metadata
    """
    print(f"Chunking audio by silence detection...")
    audio = AudioSegment.from_file(file_path)
    
    # Split on silence
    audio_segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    
    chunks = []
    current_pos = 0
    
    for i, segment in enumerate(audio_segments):
        duration = len(segment) / 1000  # convert ms to seconds
        
        # Create chunk data with metadata
        chunk_data = {
            "audio_segment": segment,
            "start_time": current_pos,
            "end_time": current_pos + duration,
            "chunk_id": i
        }
        
        chunks.append(chunk_data)
        current_pos += duration
    
    print(f"Created {len(chunks)} audio chunks based on silence detection")
    return chunks

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        The directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory 