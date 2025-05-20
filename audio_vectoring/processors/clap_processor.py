import os
import numpy as np
import torch
import time
import uuid
import tempfile
import soundfile as sf
from typing import List, Dict, Any, Optional, Union, Tuple

from qdrant_client.http import models

from audio_vectoring.processors.base import BaseAudioProcessor
from audio_vectoring.utils.audio_utils import load_audio, ensure_dir, chunk_audio_fixed_size, chunk_audio_by_silence
from audio_vectoring.chunking.audio_chunking import chunk_audio_by_semantic_shift
from audio_vectoring.embeddings.clap_embedding import ClapAudioLoader, ClapEmbeddingFunction
from audio_vectoring.storage.qdrant_connector import QdrantConnector

class ClapProcessor(BaseAudioProcessor):
    """
    Audio processor using CLAP (Contrastive Language-Audio Pretraining) for multimodal embeddings.
    Supports both text-to-audio and audio-to-audio search.
    """
    
    def __init__(self, 
                 model_name: str = "laion/larger_clap_general",
                 collection_name: str = "clap_audio_collection"):
        """
        Initialize CLAP-based audio processor
        
        Args:
            model_name: CLAP model name to use
            collection_name: Name of the Qdrant collection to use
        """
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize embedding function and data loader
        self.audio_loader = ClapAudioLoader()
        self.embedding_function = ClapEmbeddingFunction(model_name=model_name, device=self.device)
        
        # Initialize Qdrant for vector storage
        self.collection_name = collection_name
        vector_size = 512  # Size of CLAP embeddings
        self.qdrant_connector = QdrantConnector(
            collection_name=self.collection_name,
            vector_size=vector_size,
            in_memory=True
        )
        self.qdrant_client = self.qdrant_connector.client
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return as numpy array"""
        return load_audio(file_path, target_sr=48000)
    
    def vectorize_and_store(self, audio_chunks: List[Dict[str, Any]], metadatas: Optional[List[Dict]] = None) -> None:
        """Vectorize and store audio chunks in Qdrant"""
        print("Vectorizing and storing audio chunks in Qdrant...")
        
        if metadatas is None:
            metadatas = [{"chunk_id": i} for i in range(len(audio_chunks))]
        
        # Save audio chunks to temporary files for CLAP processing
        temp_files = []
        for i, chunk in enumerate(audio_chunks):
            temp_path = f"temp_clap_chunk_{i}.wav"
            sf.write(temp_path, chunk["audio"], 48000)
            temp_files.append(temp_path)
        
        # Load audio with CLAP loader
        audio_data = self.audio_loader(temp_files)
        
        # Generate embeddings
        embeddings = self.embedding_function(audio_data)
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)
        
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
    
    def search_by_text(self, query_text: str, k: int = 3) -> List[Dict]:
        """Search for audio segments using text query"""
        print(f"Searching for audio similar to text: '{query_text}'")
        
        # Generate text embedding
        text_embedding = self.embedding_function([query_text])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=text_embedding,
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
    
    def search_by_audio(self, query_audio: np.ndarray, k: int = 3) -> List[Dict]:
        """Search for audio segments using audio query"""
        print("Searching for similar audio...")
        
        # Save query audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, query_audio, 48000)
            
            # Load with audio loader
            audio_data = self.audio_loader([tmp.name])[0]
            
            # Generate embedding
            audio_embedding = self.embedding_function([audio_data])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=audio_embedding,
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
    
    def search(self, query: Union[str, np.ndarray], k: int = 3) -> List[Dict]:
        """
        Search for relevant audio segments based on query
        
        Args:
            query: Text string or audio array
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if isinstance(query, str):
            return self.search_by_text(query, k)
        else:
            return self.search_by_audio(query, k)
    
    def process_audio(self, file_path: str, chunking_method: str = "fixed", chunk_param: int = 5) -> Dict[str, Any]:
        """
        Process audio end-to-end: load, chunk, embed, store
        
        Args:
            file_path: Path to audio file
            chunking_method: Method to use for chunking ('fixed', 'silence', 'semantic')
            chunk_param: Parameter for chunking (seconds for fixed, ms for silence, % for semantic)
            
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
                
        elif chunking_method == "semantic":
            # Create a simple embedding function for semantic chunking
            def simple_embed(temp_file):
                audio_data = self.audio_loader([temp_file])[0]
                if audio_data:
                    return self.embedding_function([audio_data])[0]
                return None
            
            # Chunk by semantic shifts
            chunks = chunk_audio_by_semantic_shift(
                audio, 
                sr, 
                embedding_function=simple_embed,
                threshold=chunk_param/100  # Convert percentage to decimal
            )
            
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
        
        results["chunk_count"] = len(chunks)
        
        # Vectorize and store
        self.vectorize_and_store(chunks)
        
        # Calculate processing time
        end_time = time.time()
        results["processing_time"] = end_time - start_time
        
        return results
    
    def test_search_by_text(self, query_text: str) -> List[Dict]:
        """Test text-to-audio search with a specific query"""
        print(f"\n===== CLAP TEXT-TO-AUDIO SEARCH =====")
        print(f"Query: '{query_text}'")
        
        try:
            results = self.search_by_text(query_text, k=2)
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
            
    def test_search_by_audio(self, query_audio: np.ndarray) -> List[Dict]:
        """Test audio-to-audio search with a specific audio query"""
        print(f"\n===== CLAP AUDIO-TO-AUDIO SEARCH =====")
        print("Using audio segment as query")
        
        try:
            results = self.search_by_audio(query_audio, k=2)
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
    
    def test_search(self, query: Union[str, np.ndarray]) -> List[Dict]:
        """Test search with the appropriate method based on query type"""
        if isinstance(query, str):
            return self.test_search_by_text(query)
        else:
            return self.test_search_by_audio(query) 