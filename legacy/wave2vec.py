import os
import numpy as np
import torch
import librosa
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional, Union
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydub.silence import split_on_silence
import warnings
import soundfile as sf
from evaluate import load

# Suppress warnings
warnings.filterwarnings("ignore")

class Wav2VecProcessor:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Initialize Wav2Vec-based audio processor
        
        Args:
            model_name: Wav2Vec model name to use
        """
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"
        
# Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(in_memory=True)
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "wav2vec_audio_collection"
        
        # Create collection
        qdrant_connector.create_wav2vec_collection(self.collection_name)
        
        # Initialize metrics loader (for audio similarity evaluation)
        self.wer_metric = load("wer")
    
    def load_audio(self, file_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load audio file and return as numpy array with target sample rate"""
        print(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    
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
    
    def chunk_audio_by_spectrogram(self, audio: np.ndarray, sr: int, segment_ms: int = 1000) -> List[np.ndarray]:
        """Chunk audio by energy levels in spectrogram"""
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
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} audio chunks based on spectrogram energy")
        return chunks
    
    def vectorize_and_store_audio_documents(self, audio_chunks: List[np.ndarray], metadatas: List[Dict] = None) -> None:
        """Vectorize and store audio chunks in Qdrant"""
        print("Vectorizing and storing audio chunks in Qdrant...")
        
        if metadatas is None:
            metadatas = [{"chunk_id": i} for i in range(len(audio_chunks))]
        
        embeddings = []
        for chunk in audio_chunks:
            embedding = self.embed_audio(chunk)
            embeddings.append(embedding)
        
        # Create documents (using chunk_id as content for reference)
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
        
        print(f"Stored {len(embeddings)} audio embeddings in Qdrant collection '{self.collection_name}'")
    
    def search(self, query_audio: np.ndarray, k: int = 3) -> List[Dict]:
        """Search for relevant audio chunks based on query audio"""
        print(f"Searching for similar audio...")
        
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
            chunking_method: Method to use for chunking ('fixed', 'silence', 'spectrogram')
            chunk_param: Parameter to use for chunking (duration in seconds for fixed, 
                        min_silence_len in ms for silence, segment_ms for spectrogram)
            
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
            
        elif chunking_method == "spectrogram":
            audio_chunks = self.chunk_audio_by_spectrogram(audio, sr, chunk_param)
            
            # Estimate start and end times
            metadatas = []
            current_pos = 0
            for i, chunk in enumerate(audio_chunks):
                start_time = current_pos / sr
                end_time = (current_pos + len(chunk)) / sr
                metadatas.append({
                    "chunk_id": i,
                    "start_time": start_time,
                    "end_time": end_time
                })
                current_pos += len(chunk)
        
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
        
        # Vectorize and store
        self.vectorize_and_store_audio_documents(audio_chunks, metadatas)
        
        # Record results
        results["chunk_count"] = len(audio_chunks)
        
        # Calculate processing time
        end_time = time.time()
        results["processing_time"] = end_time - start_time
        
        return results

    def run_benchmark(self, file_path: str, query_audio_paths: List[Dict[str, Union[str, List[int]]]]):
        """
        Run full benchmark on audio file with different chunking methods
        
        Args:
            file_path: Path to audio file
            query_audio_paths: List of dicts with 'audio_path' and 'expected_chunk_ids' keys
        """
        chunking_methods = [
            {"name": "fixed", "param": 30},         # 30 second chunks
            {"name": "silence", "param": 500},      # 500ms silence detection
            {"name": "spectrogram", "param": 1000}, # 1000ms segment size for spectrogram analysis
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
            
            # Evaluate search for each query
            search_metrics = []
            for query_data in query_audio_paths:
                query_path = query_data["audio_path"]
                expected_chunk_ids = query_data["expected_chunk_ids"]
                
                # Load query audio
                query_audio, _ = self.load_audio(query_path)
                
                # Perform search
                search_results = self.search(query_audio, k=3)
                
                # Evaluate results
                query_metrics = self.evaluate_search_quality(
                    query_results=search_results,
                    expected_chunk_ids=expected_chunk_ids
                )
                
                search_metrics.append({
                    "query_path": query_path,
                    "metrics": query_metrics,
                    "search_results": search_results
                })
            
            # Aggregate results
            result = {
                "chunking_method": method["name"],
                "chunk_param": method["param"],
                "search_metrics": search_metrics,
                "processing_time": processing_results["processing_time"],
                "chunk_count": processing_results["chunk_count"]
            }
            
            results.append(result)
            
            # Clear collection for next iteration
            self.qdrant_client.delete_collection(self.collection_name)
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )
        
        # Display and return results
        self.display_benchmark_results(results)
        return results
    
    def display_benchmark_results(self, results: List[Dict]) -> None:
        """Display benchmark results in a structured format"""
        print("\n" + "="*80)
        print("WAV2VEC APPROACH BENCHMARK RESULTS")
        print("="*80)
        
        # Create DataFrame for comparison
        summary_data = []
        
        for result in results:
            # Calculate average search metrics
            avg_precision = np.mean([m["metrics"]["precision"] for m in result["search_metrics"]])
            avg_recall = np.mean([m["metrics"]["recall"] for m in result["search_metrics"]])
            avg_f1 = np.mean([m["metrics"]["f1_score"] for m in result["search_metrics"]])
            
            summary_data.append({
                "Chunking Method": result["chunking_method"],
                "Chunk Parameter": result["chunk_param"],
                "Chunk Count": result["chunk_count"],
                "Processing Time (s)": round(result["processing_time"], 2),
                "Avg Precision": round(avg_precision, 4),
                "Avg Recall": round(avg_recall, 4),
                "Avg F1 Score": round(avg_f1, 4)
            })
        
        # Create and display DataFrame
        df = pd.DataFrame(summary_data)
        print("\nSummary Metrics:")
        print(df.to_string(index=False))
        
        # Detailed results for each query
        print("\nDetailed Search Results:")
        for result in results:
            print(f"\nChunking Method: {result['chunking_method']}")
            for query_result in result["search_metrics"]:
                print(f"  Query Audio: '{query_result['query_path']}'")
                print(f"  Precision: {round(query_result['metrics']['precision'], 4)}")
                print(f"  Recall: {round(query_result['metrics']['recall'], 4)}")
                print(f"  F1 Score: {round(query_result['metrics']['f1_score'], 4)}")
                top_results = [f"Chunk {r['metadata']['chunk_id']} (score: {round(r['score'], 4)})" for r in query_result["search_results"][:2]]
                print(f"  Top Results: {top_results}")
                print()

    def test_search(self, query_audio: np.ndarray):
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

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = Wav2VecProcessor()
    
    # Use test audio file
    audio_file = "test_audio.wav"
    
    # Run benchmark or demo?
    RUN_BENCHMARK = False
    
    if RUN_BENCHMARK:
        print("\n===== WAV2VEC BENCHMARK =====")
        
        # Load audio file
        print("Loading audio file...")
        audio, sr = processor.load_audio(audio_file)
        audio_duration = len(audio) / sr
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Test different chunking methods
        chunking_methods = [
            {"name": "fixed", "param": 2},    # 2-second chunks
            {"name": "fixed", "param": 5},    # 5-second chunks
            {"name": "fixed", "param": 10},   # 10-second chunks
            {"name": "spectrogram", "param": 1000}  # Spectrogram-based
        ]
        
        # Benchmark results
        results = []
        
        for method in chunking_methods:
            start_time = time.time()
            
            if method["name"] == "fixed":
                chunks = processor.chunk_audio_fixed_size(audio, sr, chunk_duration_sec=method["param"])
                method_name = f"Fixed ({method['param']} sec)"
            elif method["name"] == "spectrogram":
                chunks = processor.chunk_audio_by_spectrogram(audio, sr, segment_ms=method["param"])
                method_name = f"Spectrogram ({method['param']} ms)"
            else:
                continue  # Skip unsupported methods
            
            # Generate embeddings for first chunk to measure embedding time
            if chunks:
                embed_start = time.time()
                embedding = processor.embed_audio(chunks[0])
                embed_time = time.time() - embed_start
                embedding_shape = embedding.shape
            else:
                embed_time = 0
                embedding_shape = (0,)
            
            end_time = time.time()
            
            results.append({
                "method": method_name,
                "chunks": len(chunks),
                "chunk_time": end_time - start_time - embed_time,
                "embed_time": embed_time,
                "total_time": end_time - start_time,
                "embedding_shape": embedding_shape
            })
        
        # Print results as a table
        print("\nWav2Vec Chunking Results:")
        print(f"{'Method':<25} {'Chunks':<10} {'Chunk Time (s)':<15} {'Embed Time (s)':<15} {'Total Time (s)':<15}")
        print("-" * 80)
        for result in results:
            print(f"{result['method']:<25} {result['chunks']:<10} {result['chunk_time']:<15.3f} {result['embed_time']:<15.3f} {result['total_time']:<15.3f}")
        
        print(f"\nEmbedding shape: {results[0]['embedding_shape']}")
    else:
        print("\n===== WAV2VEC AUDIO PROCESSING =====")
        # Load audio file
        print("Loading audio file...")
        audio, sr = processor.load_audio(audio_file)
        audio_duration = len(audio) / sr
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Process with fixed-size chunking for demonstration
        print("\nChunking audio into fixed-size segments...")
        audio_chunks = processor.chunk_audio_fixed_size(audio, sr, chunk_duration_sec=5)
        print(f"Created {len(audio_chunks)} audio chunks")
        
        # Embed first chunk as example
        if audio_chunks:
            print("\nGenerating embedding for first chunk...")
            embedding = processor.embed_audio(audio_chunks[0])
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 5 values: {embedding[:5]}")
            
            # Create sample search with same chunk
            print("\n===== AUDIO SIMILARITY SEARCH DEMO =====")
            print("Setting up demo similarity search...")
            
            # Store chunks in vector DB
            metadatas = [{"chunk_id": i} for i in range(len(audio_chunks))]
            processor.vectorize_and_store_audio_documents(audio_chunks[:5], metadatas[:5])
            
            # Search using the first chunk as query
            print("\nSearching for similar audio segments...")
            search_results = processor.search(audio_chunks[0], k=2)
            
            # Display results
            print("\nSearch results:")
            for i, result in enumerate(search_results):
                print(f"Result {i+1}: Chunk {result['metadata']['chunk_id']} (score: {result['score']:.4f})")
    
    print("\n===== WAV2VEC PROCESSING COMPLETED SUCCESSFULLY! =====")
    
    # Test search with a segment of the audio
    if not RUN_BENCHMARK:
        audio, sr = processor.load_audio(audio_file)
        query_segment = audio[10*sr:15*sr]  # 5-second segment starting at 10s
        processor.test_search(query_segment)
