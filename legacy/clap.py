import os
import numpy as np
import torch
import librosa
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from pydub import AudioSegment
from transformers import ClapModel, ClapProcessor as HFClapProcessor
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydub.silence import split_on_silence
import warnings
import soundfile as sf
from tqdm import tqdm
import json
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore")

class ClapAudioLoader:
    """
    Audio loader class that loads and processes audio files for CLAP embedding
    """
    def __init__(self, target_sample_rate: int = 48000) -> None:
        self.target_sample_rate = target_sample_rate

    def load_audio(self, uri: str) -> Optional[Dict[str, Any]]:
        """Load audio file from URI"""
        if uri is None:
            return None

        try:
            waveform, sample_rate = librosa.load(uri, sr=self.target_sample_rate, mono=True)
            return {"waveform": waveform, "uri": uri}
        except Exception as e:
            print(f"Error loading audio file {uri}: {str(e)}")
            return None

    def __call__(self, uris: List[Optional[str]]) -> List[Optional[Dict[str, Any]]]:
        """Process multiple URIs"""
        return [self.load_audio(uri) for uri in uris]


class ClapEmbeddingFunction:
    """
    Embedding function for CLAP that can embed both audio and text
    """
    def __init__(
        self,
        model_name: str = "laion/larger_clap_general",
        device: str = None
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading CLAP model from {model_name} on {device}...")
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = HFClapProcessor.from_pretrained(model_name)
        self.device = device
        print("Model loaded successfully")

    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio using CLAP"""
        inputs = self.processor(audios=audio, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_embedding = self.model.get_audio_features(**inputs)
        
        return audio_embedding.squeeze().cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLAP"""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        
        return text_embedding.squeeze().cpu().numpy()

    def __call__(self, input_data: Union[List[str], List[Dict[str, Any]]]) -> List[np.ndarray]:
        """Process either text or audio inputs"""
        embeddings = []
        
        for item in input_data:
            if isinstance(item, dict) and "waveform" in item:
                embeddings.append(self.encode_audio(item["waveform"]))
            elif isinstance(item, str):
                embeddings.append(self.encode_text(item))
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        
        return embeddings


class ClapProcessor:
    def __init__(self, model_name: str = "laion/larger_clap_general"):
        """
        Initialize CLAP-based audio processor
        
        Args:
            model_name: CLAP model name to use
        """
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize embedding function and data loader
        self.audio_loader = ClapAudioLoader()
        self.embedding_function = ClapEmbeddingFunction(model_name=model_name, device=self.device)
        
# Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(in_memory=True)
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "clap_audio_collection"
        
        # Create collection
        qdrant_connector.create_clap_collection(self.collection_name)
    
    def load_audio(self, file_path: str, target_sr: int = 48000) -> np.ndarray:
        """Load audio file and return as numpy array with target sample rate"""
        print(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    
    def chunk_audio_fixed_size(self, audio: np.ndarray, sr: int, chunk_duration_sec: int = 30) -> List[np.ndarray]:
        """Chunk audio into fixed-size segments"""
        print(f"Chunking audio into fixed {chunk_duration_sec}-second segments...")
        chunk_size = sr * chunk_duration_sec
        chunks = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) > 0.5 * chunk_size:  # Only keep chunks that are at least half the desired size
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} fixed-size audio chunks")
        return chunks
    
    def chunk_audio_by_silence(self, file_path: str, min_silence_len: int = 500, silence_thresh: int = -40) -> List[AudioSegment]:
        """Chunk audio by detecting silence using pydub"""
        print(f"Chunking audio by silence detection...")
        audio = AudioSegment.from_file(file_path)
        
        # Split on silence
        audio_chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        print(f"Created {len(audio_chunks)} audio chunks based on silence detection")
        return audio_chunks
    
    def chunk_audio_by_semantic_shift(self, audio: np.ndarray, sr: int, window_sec: int = 5, step_sec: float = 2.5, threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Chunk audio by detecting semantic shifts in embeddings"""
        print(f"Chunking audio by semantic shifts...")
        
        # Parameters
        window_samples = int(window_sec * sr)
        step_samples = int(step_sec * sr)
        
        # Generate sliding windows
        windows = []
        for i in range(0, len(audio) - window_samples + 1, step_samples):
            window = audio[i:i + window_samples]
            windows.append({
                "audio": window,
                "start_sample": i,
                "end_sample": i + window_samples,
                "start_time": i / sr,
                "end_time": (i + window_samples) / sr
            })
        
        # If no valid windows, return empty list
        if not windows:
            print("No valid windows found")
            return []
        
        # Compute embeddings for each window
        print(f"Computing embeddings for {len(windows)} windows...")
        embeddings = []
        
        for window in windows:
            # We need to save the audio temporarily to use the audio loader
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, window["audio"], sr)
                
                # Load with audio loader and embed
                audio_data = self.audio_loader([tmp.name])[0]
                if audio_data:
                    embedding = self.embedding_function([audio_data])[0]
                    window["embedding"] = embedding
                    embeddings.append(embedding)
        
        # Compute cosine similarity between consecutive windows
        boundaries = [0]  # Always include the first window
        
        for i in range(1, len(embeddings)):
            # Compute cosine similarity
            sim = np.dot(embeddings[i-1], embeddings[i]) / (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]))
            
            # If similarity drops below threshold, mark as a boundary
            if sim < (1.0 - threshold):
                boundaries.append(i)
        
        # Always include the last window
        if len(embeddings) - 1 not in boundaries:
            boundaries.append(len(embeddings) - 1)
        
        # Create chunks from boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            # Get the audio range
            start_sample = windows[start_idx]["start_sample"]
            end_sample = windows[end_idx]["end_sample"]
            
            # Extract the chunk
            chunk = {
                "audio": audio[start_sample:end_sample],
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time": start_sample / sr,
                "end_time": end_sample / sr
            }
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} audio chunks based on semantic shifts")
        return chunks
    
    def vectorize_and_store_audio_chunks(self, audio_chunks: List[np.ndarray], metadatas: List[Dict] = None) -> None:
        """Vectorize and store audio chunks in Qdrant"""
        print("Vectorizing and storing audio chunks in Qdrant...")
        
        if metadatas is None:
            metadatas = [{"chunk_id": i} for i in range(len(audio_chunks))]
        
        # Create temporary files for the audio chunks
        temp_files = []
        for i, chunk in enumerate(audio_chunks):
            temp_path = f"temp_chunk_{i}.wav"
            sf.write(temp_path, chunk, 48000)
            temp_files.append(temp_path)
        
        # Load audio chunks using the audio loader
        loaded_chunks = self.audio_loader(temp_files)
        
        # Generate embeddings
        embeddings = self.embedding_function(loaded_chunks)
        
        # Create documents
        documents = [
            Document(page_content=f"Audio chunk {meta['chunk_id']}", metadata=meta)
            for meta in metadatas
        ]
        
        # Store in Qdrant with proper UUIDs
        import uuid
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=[str(uuid.uuid4()) for _ in range(len(embeddings))],
                vectors=embeddings,
                payloads=[{"document": documents[i].page_content, "metadata": documents[i].metadata} 
                          for i in range(len(embeddings))]
            )
        )
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)
        
        print(f"Stored {len(embeddings)} audio embeddings in Qdrant collection '{self.collection_name}'")
    
    def search_by_text(self, query_text: str, k: int = 3) -> List[Dict]:
        """Search for relevant audio chunks based on text query"""
        print(f"Searching for audio similar to text: '{query_text}'")
        
        # Generate text embedding
        query_embedding = self.embedding_function([query_text])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        # Format results
        results = []
        for res in search_results:
            results.append({
                "id": res.id,
                "score": res.score,
                "document": res.payload["document"],
                "metadata": res.payload["metadata"]
            })
        
        return results
    
    def search_by_audio(self, query_audio: np.ndarray, k: int = 3) -> List[Dict]:
        """Search for relevant audio chunks based on audio query"""
        print(f"Searching for similar audio...")
        
        # Save query audio to temporary file
        temp_path = "temp_query.wav"
        sf.write(temp_path, query_audio, 48000)
        
        # Load with audio loader
        loaded_query = self.audio_loader([temp_path])[0]
        
        # Generate embedding
        query_embedding = self.embedding_function([loaded_query])[0]
        
        # Clean up
        os.remove(temp_path)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        # Format results
        results = []
        for res in search_results:
            results.append({
                "id": res.id,
                "score": res.score,
                "document": res.payload["document"],
                "metadata": res.payload["metadata"]
            })
        
        return results
    
    def evaluate_search_quality(self, query_results: List[Dict], expected_chunk_ids: List[int]) -> Dict[str, float]:
        """Evaluate search quality based on expected chunk IDs"""
        retrieved_ids = [int(result["metadata"]["chunk_id"]) for result in query_results]
        
        # Precision: How many of the results are relevant
        precision = len(set(retrieved_ids) & set(expected_chunk_ids)) / len(retrieved_ids) if retrieved_ids else 0
        
        # Recall: How many of the relevant chunks were retrieved
        recall = len(set(retrieved_ids) & set(expected_chunk_ids)) / len(expected_chunk_ids) if expected_chunk_ids else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def process_audio(self, file_path: str, chunking_method: str = "fixed", chunk_param: int = 30) -> Dict:
        """
        Process audio end-to-end: load, chunk, embed, store
        
        Args:
            file_path: Path to audio file
            chunking_method: Method to use for chunking ('fixed', 'silence', 'semantic')
            chunk_param: Parameter to use for chunking (duration in seconds for fixed, 
                        min_silence_len in ms for silence, threshold for semantic)
            
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
            audio_chunks = self.chunk_audio_fixed_size(audio, sr, chunk_param)
            metadatas = [{"chunk_id": i, "start_time": i * chunk_param, "end_time": (i + 1) * chunk_param} 
                         for i in range(len(audio_chunks))]
            
        elif chunking_method == "silence":
            # Get chunks by silence
            pydub_chunks = self.chunk_audio_by_silence(file_path, min_silence_len=chunk_param)
            
            # Convert pydub chunks to numpy arrays
            audio_chunks = []
            metadatas = []
            current_time = 0
            
            for i, chunk in enumerate(pydub_chunks):
                # Save chunk temporarily
                temp_path = f"temp_chunk_{i}.wav"
                chunk.export(temp_path, format="wav")
                
                # Load as numpy
                chunk_audio, chunk_sr = self.load_audio(temp_path)
                audio_chunks.append(chunk_audio)
                
                # Update metadata
                chunk_duration = len(chunk) / 1000  # pydub duration is in milliseconds
                metadatas.append({
                    "chunk_id": i,
                    "start_time": current_time,
                    "end_time": current_time + chunk_duration
                })
                current_time += chunk_duration
                
                # Clean up temp file
                os.remove(temp_path)
            
        elif chunking_method == "semantic":
            # Use semantic shift detection for chunking
            semantic_chunks = self.chunk_audio_by_semantic_shift(
                audio, sr, window_sec=5, step_sec=2.5, threshold=chunk_param/100.0
            )
            
            # Extract audio and create metadata
            audio_chunks = [chunk["audio"] for chunk in semantic_chunks]
            metadatas = [
                {
                    "chunk_id": i,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"]
                }
                for i, chunk in enumerate(semantic_chunks)
            ]
        
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
        
        # Vectorize and store
        self.vectorize_and_store_audio_chunks(audio_chunks, metadatas)
        
        # Record results
        results["chunk_count"] = len(audio_chunks)
        
        # Calculate processing time
        end_time = time.time()
        results["processing_time"] = end_time - start_time
        
        return results
    
    def run_benchmark(self, file_path: str, text_queries: List[Dict[str, Union[str, List[int]]]], audio_queries: List[Dict[str, Union[str, List[int]]]] = None):
        """
        Run full benchmark on audio file with different chunking methods
        
        Args:
            file_path: Path to audio file
            text_queries: List of dicts with 'query' and 'expected_chunk_ids' keys
            audio_queries: Optional list of dicts with 'audio_path' and 'expected_chunk_ids' keys
        """
        chunking_methods = [
            {"name": "fixed", "param": 30},     # 30 second chunks
            {"name": "silence", "param": 500},  # 500ms silence detection
            {"name": "semantic", "param": 25},  # 0.25 threshold for semantic shift detection
        ]
        
        results = []
        
        for method in chunking_methods:
            print(f"\n{'-'*50}")
            print(f"Running benchmark with {method['name']} chunking")
            print(f"{'-'*50}")
            
            # Process audio
            processing_results = self.process_audio(
                file_path=file_path,
                chunking_method=method["name"],
                chunk_param=method["param"]
            )
            
            # Evaluate text search queries
            text_search_metrics = []
            for query_data in text_queries:
                query_text = query_data["query"]
                expected_chunk_ids = query_data["expected_chunk_ids"]
                
                # Perform search
                search_results = self.search_by_text(query_text, k=3)
                
                # Evaluate results
                query_metrics = self.evaluate_search_quality(
                    query_results=search_results,
                    expected_chunk_ids=expected_chunk_ids
                )
                
                text_search_metrics.append({
                    "query": query_text,
                    "metrics": query_metrics,
                    "search_results": search_results
                })
            
            # Evaluate audio search queries if provided
            audio_search_metrics = []
            if audio_queries:
                for query_data in audio_queries:
                    query_path = query_data["audio_path"]
                    expected_chunk_ids = query_data["expected_chunk_ids"]
                    
                    # Load query audio
                    query_audio, _ = self.load_audio(query_path)
                    
                    # Perform search
                    search_results = self.search_by_audio(query_audio, k=3)
                    
                    # Evaluate results
                    query_metrics = self.evaluate_search_quality(
                        query_results=search_results,
                        expected_chunk_ids=expected_chunk_ids
                    )
                    
                    audio_search_metrics.append({
                        "query_path": query_path,
                        "metrics": query_metrics,
                        "search_results": search_results
                    })
            
            # Aggregate results
            result = {
                "chunking_method": method["name"],
                "chunk_param": method["param"],
                "text_search_metrics": text_search_metrics,
                "audio_search_metrics": audio_search_metrics,
                "processing_time": processing_results["processing_time"],
                "chunk_count": processing_results["chunk_count"]
            }
            
            results.append(result)
            
            # Clear collection for next iteration
            self.qdrant_client.delete_collection(self.collection_name)
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE,
                ),
            )
        
        # Display and return results
        self.display_benchmark_results(results)
        return results
    
    def display_benchmark_results(self, results: List[Dict]) -> None:
        """Display benchmark results in a structured format"""
        print("\n" + "="*80)
        print("CLAP APPROACH BENCHMARK RESULTS")
        print("="*80)
        
        # Create DataFrame for comparison
        summary_data = []
        
        for result in results:
            # Calculate average search metrics for text queries
            text_avg_precision = np.mean([m["metrics"]["precision"] for m in result["text_search_metrics"]])
            text_avg_recall = np.mean([m["metrics"]["recall"] for m in result["text_search_metrics"]])
            text_avg_f1 = np.mean([m["metrics"]["f1_score"] for m in result["text_search_metrics"]])
            
            # Calculate average search metrics for audio queries
            audio_avg_precision = 0
            audio_avg_recall = 0
            audio_avg_f1 = 0
            
            if result["audio_search_metrics"]:
                audio_avg_precision = np.mean([m["metrics"]["precision"] for m in result["audio_search_metrics"]])
                audio_avg_recall = np.mean([m["metrics"]["recall"] for m in result["audio_search_metrics"]])
                audio_avg_f1 = np.mean([m["metrics"]["f1_score"] for m in result["audio_search_metrics"]])
            
            # Record metrics
            record = {
                "Chunking Method": result["chunking_method"],
                "Chunk Parameter": result["chunk_param"],
                "Chunk Count": result["chunk_count"],
                "Processing Time (s)": round(result["processing_time"], 2),
                "Text Avg Precision": round(text_avg_precision, 4),
                "Text Avg Recall": round(text_avg_recall, 4),
                "Text Avg F1 Score": round(text_avg_f1, 4)
            }
            
            # Add audio metrics if available
            if result["audio_search_metrics"]:
                record.update({
                    "Audio Avg Precision": round(audio_avg_precision, 4),
                    "Audio Avg Recall": round(audio_avg_recall, 4),
                    "Audio Avg F1 Score": round(audio_avg_f1, 4)
                })
                
            summary_data.append(record)
        
        # Create and display DataFrame
        df = pd.DataFrame(summary_data)
        print("\nSummary Metrics:")
        print(df.to_string(index=False))
        
        # Detailed results for text queries
        print("\nDetailed Text Search Results:")
        for result in results:
            print(f"\nChunking Method: {result['chunking_method']}")
            for query_result in result["text_search_metrics"]:
                print(f"  Query: '{query_result['query']}'")
                print(f"  Precision: {round(query_result['metrics']['precision'], 4)}")
                print(f"  Recall: {round(query_result['metrics']['recall'], 4)}")
                print(f"  F1 Score: {round(query_result['metrics']['f1_score'], 4)}")
                top_results = [f"Chunk {r['metadata']['chunk_id']} (score: {round(r['score'], 4)})" for r in query_result["search_results"][:2]]
                print(f"  Top Results: {top_results}")
                print()
        
        # Detailed results for audio queries
        if any(result["audio_search_metrics"] for result in results):
            print("\nDetailed Audio Search Results:")
            for result in results:
                if result["audio_search_metrics"]:
                    print(f"\nChunking Method: {result['chunking_method']}")
                    for query_result in result["audio_search_metrics"]:
                        print(f"  Query Audio: '{query_result['query_path']}'")
                        print(f"  Precision: {round(query_result['metrics']['precision'], 4)}")
                        print(f"  Recall: {round(query_result['metrics']['recall'], 4)}")
                        print(f"  F1 Score: {round(query_result['metrics']['f1_score'], 4)}")
                        top_results = [f"Chunk {r['metadata']['chunk_id']} (score: {round(r['score'], 4)})" for r in query_result["search_results"][:2]]
                        print(f"  Top Results: {top_results}")
                        print()

    def test_search_by_text(self, query_text: str):
        """Test search with a specific text query"""
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
            
    def test_search_by_audio(self, query_audio: np.ndarray):
        """Test search with a specific audio query"""
        print(f"\n===== CLAP AUDIO-TO-AUDIO SEARCH =====")
        print("Using audio segment as query")
        
        try:
            results = self.search_by_audio(query_audio, k=2)
            print(f"\nResults:")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: Segment {result['metadata']['chunk_id']} (score: {result['score']:.4f})")
                if "start_time" in result["metadata"] and "end_time" in result["metadata"]:
                    print(f"    Time range: {result['metadata']['start_time']:.2f}s - {result['metadata']['end_time']:.2f}s")
            return results
        except Exception as e:
            print(f"  Search failed: {e}")
            return []
        
# Example usage
if __name__ == "__main__":
    # Initialize CLAP processor
    processor = ClapProcessor()
    
    # Use test audio file
    audio_file = "test_audio.wav"
    
    # Run benchmark or demo?
    RUN_BENCHMARK = False
    
    if RUN_BENCHMARK:
        print("\n===== CLAP BENCHMARK =====")
        
        # Load audio file
        print("Loading audio file...")
        audio, sr = processor.load_audio(audio_file)
        audio_duration = len(audio) / sr
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Test different chunking methods
        chunking_methods = [
            {"name": "fixed", "param": 5},     # 5-second chunks
            {"name": "fixed", "param": 10},    # 10-second chunks
            {"name": "semantic", "param": 25}  # Semantic chunking with 0.25 threshold
        ]
        
        # Benchmark results
        results = []
        
        for method in chunking_methods:
            start_time = time.time()
            
            if method["name"] == "fixed":
                chunks = processor.chunk_audio_fixed_size(audio, sr, chunk_duration_sec=method["param"])
                method_name = f"Fixed ({method['param']} sec)"
            elif method["name"] == "semantic":
                try:
                    chunks = processor.chunk_audio_by_semantic_shift(audio, sr, threshold=method["param"]/100)
                    method_name = f"Semantic (threshold={method['param']}%)"
                except Exception as e:
                    print(f"Error with semantic chunking: {e}")
                    chunks = []
                    method_name = f"Semantic (failed)"
            else:
                continue  # Skip unsupported methods
            
            # Measure embedding time for text and audio
            if chunks and len(chunks) > 0:
                # For fixed chunking, we have direct audio arrays
                if method["name"] == "fixed":
                    chunk_sample = chunks[0]
                # For semantic chunking, we have dictionaries with 'audio' key
                elif method["name"] == "semantic" and len(chunks) > 0:
                    chunk_sample = chunks[0]["audio"]
                else:
                    chunk_sample = None
                
                if chunk_sample is not None:
                    # Measure audio embedding time
                    embed_start = time.time()
                    
                    # Save audio to temp file for embedding
                    temp_path = "temp_benchmark.wav"
                    sf.write(temp_path, chunk_sample, sr)
                    
                    # Load with audio loader
                    loaded_audio = processor.audio_loader([temp_path])[0]
                    
                    # Generate embedding
                    audio_embedding = processor.embedding_function([loaded_audio])[0]
                    audio_embed_time = time.time() - embed_start
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    # Measure text embedding time
                    text_start = time.time()
                    text_embedding = processor.embedding_function(["test query for benchmark"])[0]
                    text_embed_time = time.time() - text_start
                    
                    embedding_shape = audio_embedding.shape
                else:
                    audio_embed_time = 0
                    text_embed_time = 0
                    embedding_shape = (0,)
            else:
                audio_embed_time = 0
                text_embed_time = 0
                embedding_shape = (0,)
            
            end_time = time.time()
            
            chunk_count = len(chunks)
            
            results.append({
                "method": method_name,
                "chunks": chunk_count,
                "chunk_time": end_time - start_time - audio_embed_time - text_embed_time,
                "audio_embed_time": audio_embed_time,
                "text_embed_time": text_embed_time,
                "total_time": end_time - start_time,
                "embedding_shape": embedding_shape
            })
        
        # Print results as a table
        print("\nCLAP Chunking Results:")
        print(f"{'Method':<25} {'Chunks':<10} {'Chunk Time (s)':<15} {'Audio Embed (s)':<15} {'Text Embed (s)':<15} {'Total (s)':<10}")
        print("-" * 90)
        for result in results:
            print(f"{result['method']:<25} {result['chunks']:<10} {result['chunk_time']:<15.3f} {result['audio_embed_time']:<15.3f} {result['text_embed_time']:<15.3f} {result['total_time']:<10.3f}")
        
        if results and results[0]['embedding_shape'][0] > 0:
            print(f"\nEmbedding shape: {results[0]['embedding_shape']}")
        
        print("\nKey Advantages of CLAP:")
        print("1. Multimodal capabilities - can search audio with text queries")
        print("2. Semantic understanding of audio content")
        print("3. More robust to background noise and audio quality variations")
    else:
        print("\n===== CLAP AUDIO PROCESSING =====")
        # Load audio file
        print("Loading audio file...")
        audio, sr = processor.load_audio(audio_file)
        audio_duration = len(audio) / sr
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Process with fixed-size chunking for demonstration
        print("\nChunking audio into fixed-size segments...")
        audio_chunks = processor.chunk_audio_fixed_size(audio, sr, chunk_duration_sec=5)
        print(f"Created {len(audio_chunks)} audio chunks")
        
        # Store first few chunks in vector DB
        if audio_chunks:
            print("\n===== VECTOR STORAGE =====")
            print("Storing audio chunks in vector database...")
            chunk_subset = audio_chunks[:min(3, len(audio_chunks))]  # Use up to 3 chunks
            metadatas = [{"chunk_id": i} for i in range(len(chunk_subset))]
            
            try:
                processor.vectorize_and_store_audio_chunks(chunk_subset, metadatas)
                
                # Demo multimodal search capability
                print("\n===== MULTIMODAL SEARCH DEMO =====")
                
                # Text-to-audio search
                test_query = "someone speaking to an audience"
                print(f"\nSearching audio using text query: '{test_query}'")
                
                try:
                    text_results = processor.search_by_text(test_query, k=2)
                    print("\nText-to-audio search results:")
                    for i, result in enumerate(text_results):
                        print(f"Result {i+1}: Chunk {result['metadata']['chunk_id']} (score: {result['score']:.4f})")
                except Exception as e:
                    print(f"Text search failed: {e}")
                
                # Audio-to-audio search (using first chunk as query)
                print("\nSearching for similar audio segments...")
                try:
                    audio_results = processor.search_by_audio(audio_chunks[0], k=2)
                    print("\nAudio-to-audio search results:")
                    for i, result in enumerate(audio_results):
                        print(f"Result {i+1}: Chunk {result['metadata']['chunk_id']} (score: {result['score']:.4f})")
                except Exception as e:
                    print(f"Audio search failed: {e}")
                    
            except Exception as e:
                print(f"Error during vector storage or search: {e}")
    
    print("\n===== CLAP PROCESSING COMPLETED SUCCESSFULLY! =====")
    
    # Test search if not in benchmark mode
    if not RUN_BENCHMARK:
        # Text search
        processor.test_search_by_text("What does the speaker say about the audience?")
        
        # Audio search
        audio, sr = processor.load_audio(audio_file)
        query_segment = audio[10*sr:15*sr]  # 5-second segment starting at 10s
        processor.test_search_by_audio(query_segment)
