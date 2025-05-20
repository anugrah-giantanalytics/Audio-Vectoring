import numpy as np
import librosa
import tempfile
import soundfile as sf
from typing import List, Dict, Any, Optional

def chunk_audio_by_spectrogram(audio: np.ndarray, sr: int, segment_ms: int = 1000) -> List[Dict[str, Any]]:
    """
    Chunk audio by energy levels in spectrogram
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        segment_ms: Minimum segment duration in milliseconds
        
    Returns:
        List of dictionaries containing audio chunks and metadata
    """
    print(f"Chunking audio by spectrogram energy...")
    
    # Compute spectrogram
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    # Convert segment_ms to frames
    segment_frames = int((segment_ms / 1000) * sr / hop_length)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Calculate energy per time frame
    energy = np.mean(mel_spec_db, axis=0)
    
    # Find segments by energy threshold
    threshold = np.mean(energy) - 0.5 * np.std(energy)
    is_above_threshold = energy > threshold
    
    # Find segment boundaries
    segment_starts = []
    current_state = False
    
    for i, state in enumerate(is_above_threshold):
        if not current_state and state:
            # Transition from below to above threshold
            segment_starts.append(i)
            current_state = True
        elif current_state and not state:
            # Transition from above to below threshold
            current_state = False
    
    # Add final segment if needed
    if len(segment_starts) == 0:
        segment_starts = [0]
    
    # Create chunks
    chunks = []
    for i, start_frame in enumerate(segment_starts):
        # Convert frame to sample index
        start_sample = start_frame * hop_length
        
        # Determine end sample
        if i < len(segment_starts) - 1:
            end_sample = segment_starts[i+1] * hop_length
        else:
            end_sample = len(audio)
        
        # Extract chunk
        chunk = audio[start_sample:end_sample]
        
        # Only add if chunk is long enough
        if len(chunk) > 0.25 * sr:  # At least 0.25 seconds
            chunk_data = {
                "audio": chunk,
                "start_time": start_sample / sr,
                "end_time": end_sample / sr,
                "chunk_id": len(chunks)
            }
            chunks.append(chunk_data)
    
    print(f"Created {len(chunks)} audio chunks based on spectrogram energy")
    return chunks

def chunk_audio_by_semantic_shift(audio: np.ndarray, sr: int, 
                                 embedding_function: Any,
                                 window_sec: int = 5, 
                                 step_sec: float = 2.5, 
                                 threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Chunk audio by detecting semantic shifts in embeddings
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        embedding_function: Function to create embeddings from audio
        window_sec: Window size in seconds
        step_sec: Step size in seconds
        threshold: Similarity threshold for detecting shifts
        
    Returns:
        List of dictionaries containing audio chunks and metadata
    """
    print(f"Chunking audio by semantic shifts...")
    
    # Parameters
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)
    
    # Generate sliding windows
    windows = []
    for i in range(0, len(audio) - window_samples + 1, step_samples):
        window = audio[i:i + window_samples]
        window_data = {
            "audio": window,
            "start_sample": i,
            "end_sample": i + window_samples,
            "start_time": i / sr,
            "end_time": (i + window_samples) / sr
        }
        windows.append(window_data)
    
    # If no valid windows, return empty list
    if not windows:
        print("No valid windows found")
        return []
    
    # Compute embeddings for each window
    print(f"Computing embeddings for {len(windows)} windows...")
    
    for i, window in enumerate(windows):
        # We need to save the audio temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, window["audio"], sr)
            
            # Generate embedding
            try:
                embedding = embedding_function(tmp.name)
                window["embedding"] = embedding
            except Exception as e:
                print(f"Error computing embedding for window {i}: {e}")
                continue
    
    # Compute cosine similarity between consecutive windows and find segment boundaries
    segment_boundaries = [0]  # Start with the first window
    
    for i in range(1, len(windows)):
        if "embedding" not in windows[i] or "embedding" not in windows[i-1]:
            continue
            
        current_embedding = windows[i]["embedding"]
        previous_embedding = windows[i-1]["embedding"]
        
        # Calculate cosine similarity
        similarity = np.dot(current_embedding, previous_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
        )
        
        # If similarity is below threshold, mark as new segment
        if similarity < (1 - threshold):
            segment_boundaries.append(i)
    
    # Always include the last window
    if len(windows) > 0 and segment_boundaries[-1] != len(windows) - 1:
        segment_boundaries.append(len(windows) - 1)
    
    # Create segments based on boundaries
    segments = []
    
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]
        
        # Get start and end times
        start_time = windows[start_idx]["start_time"]
        end_time = windows[end_idx]["end_time"]
        
        # Create segment audio by combining windows
        start_sample = windows[start_idx]["start_sample"]
        end_sample = windows[end_idx]["end_sample"]
        segment_audio = audio[start_sample:end_sample]
        
        # Create segment data
        segment_data = {
            "audio": segment_audio,
            "start_time": start_time,
            "end_time": end_time,
            "chunk_id": i
        }
        
        segments.append(segment_data)
    
    print(f"Created {len(segments)} audio segments based on semantic shifts")
    return segments 