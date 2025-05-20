import os
import numpy as np
import torch
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from qdrant_client.http import models

from audio_vectoring.processors.base import BaseAudioProcessor
from audio_vectoring.utils.audio_utils import load_audio, ensure_dir, chunk_audio_fixed_size, chunk_audio_by_silence
from audio_vectoring.chunking.audio_chunking import chunk_audio_by_spectrogram
from audio_vectoring.storage.qdrant_connector import QdrantConnector

class Wav2VecProcessor(BaseAudioProcessor):
    """
    Audio processor using Facebook's Wav2Vec2 model for direct audio embeddings.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base-960h",
                 collection_name: str = "wav2vec_audio_collection"):
        """
        Initialize Wav2Vec-based audio processor
        
        Args:
            model_name: Wav2Vec model name to use
            collection_name: Name of the Qdrant collection to use
        """
        # Initialize Wav2Vec model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Set device
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Initialize Qdrant for vector storage
        self.collection_name = collection_name
        vector_size = 768  # Size of Wav2Vec2 embeddings
        self.qdrant_connector = QdrantConnector(
            collection_name=self.collection_name,
            vector_size=vector_size,
            in_memory=True
        )
        self.qdrant_client = self.qdrant_connector.client
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return as numpy array"""
        return load_audio(file_path, target_sr=16000)
    
    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """Generate embeddings for audio using Wav2Vec2"""
        print("Embedding audio with Wav2Vec2...")
        
        # Process audio with Wav2Vec2
        input_values = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        if self.device == "cuda":
            input_values = input_values.to("cuda")
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(input_values)
            # Using the last hidden state as the embedding
            last_hidden_state = outputs.last_hidden_state
            
            # Average pooling over time dimension to get a fixed-size embedding
            embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
        
        return embedding
    
    def vectorize_and_store(self, audio_chunks: List[Dict[str, Any]], metadatas: Optional[List[Dict]] = None) -> None:
        """Vectorize and store audio chunks in Qdrant"""
        print("Vectorizing and storing audio chunks in Qdrant...")
        
        if metadatas is None:
            metadatas = [{"chunk_id": i} for i in range(len(audio_chunks))]
        
        embeddings = []
        for chunk in audio_chunks:
            embedding = self.embed_audio(chunk["audio"])
            embeddings.append(embedding)
        
        # Store in Qdrant with proper UUIDs
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=[str(uuid.uuid4()) for _ in range(len(embeddings))],
                vectors=embeddings,
                payloads=metadatas
            )
        )
        
        print(f"Stored {len(embeddings)} audio embeddings in Qdrant collection '{self.collection_name}'")
    
    def search(self, query_audio: np.ndarray, k: int = 3) -> List[Dict]:
        """Search for similar audio segments"""
        print("Searching for similar audio...")
        
        # Generate embedding for query audio
        query_embedding = self.embed_audio(query_audio)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "score": result.score,
                "metadata": result.payload
            })
        
        return results
    
    def process_audio(self, file_path: str, chunking_method: str = "fixed", chunk_param: int = 5) -> Dict[str, Any]:
        """
        Process audio end-to-end: load, chunk, embed, store
        
        Args:
            file_path: Path to audio file
            chunking_method: Method to use for chunking ('fixed', 'silence', 'spectrogram')
            chunk_param: Parameter for chunking (seconds for fixed, ms for silence, ms for spectrogram)
            
        Returns:
            Dict with processing results including metrics
        """
        start_time = time.time()
        results = {
            "file_path": file_path,
            "chunking_method": chunking_method,
            "chunk_param": chunk_param
        }
        
        # Load audio
        audio, sr = self.load_audio(file_path)
        results["audio_duration"] = len(audio) / sr
        
        # Chunk based on method
        if chunking_method == "fixed":
            # Chunk into fixed-size segments
            chunks = chunk_audio_fixed_size(audio, sr, chunk_duration_sec=chunk_param)
            
        elif chunking_method == "silence":
            # Chunk by detecting silence
            silence_chunks = chunk_audio_by_silence(file_path, min_silence_len=chunk_param)
            
            # Convert pydub AudioSegment chunks to numpy arrays
            chunks = []
            for i, segment in enumerate(silence_chunks):
                # Export to temporary file and reload as numpy
                temp_path = f"temp_chunk_{i}.wav"
                segment["audio_segment"].export(temp_path, format="wav")
                chunk_audio, chunk_sr = self.load_audio(temp_path)
                
                # Create chunk data
                chunk_data = {
                    "audio": chunk_audio,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "chunk_id": segment["chunk_id"]
                }
                chunks.append(chunk_data)
                
                # Clean up temp file
                os.remove(temp_path)
            
        elif chunking_method == "spectrogram":
            # Chunk by spectrogram energy
            chunks = chunk_audio_by_spectrogram(audio, sr, segment_ms=chunk_param)
            
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
        
        results["chunk_count"] = len(chunks)
        
        # Vectorize and store
        self.vectorize_and_store(chunks)
        
        # Calculate processing time
        end_time = time.time()
        results["processing_time"] = end_time - start_time
        
        return results
    
    def test_search(self, query_audio: np.ndarray) -> List[Dict]:
        """Test search with a specific audio query"""
        print(f"\n===== WAV2VEC AUDIO SEARCH =====")
        print("Using audio segment as query")
        
        try:
            results = self.search(query_audio, k=2)
            print(f"\nResults:")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: Segment {result['metadata']['chunk_id']} (score: {result['score']:.4f})")
                # Check if time range is available in metadata
                if "start_time" in result["metadata"] and "end_time" in result["metadata"]:
                    print(f"    Time range: {result['metadata']['start_time']:.2f}s - {result['metadata']['end_time']:.2f}s")
            return results
        except Exception as e:
            print(f"  Search failed: {e}")
            return [] 