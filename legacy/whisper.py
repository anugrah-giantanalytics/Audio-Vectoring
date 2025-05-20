import os
import numpy as np
import torch
import librosa
import pandas as pd
import time
import json
from typing import List, Dict, Tuple, Optional, Union
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import soundfile as sf
from evaluate import load

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class WhisperAudioProcessor:
    def __init__(self, model_name: str = "openai/whisper-base", api_key: Optional[str] = None):
        """
        Initialize Whisper-based audio processor
        
        Args:
            model_name: Whisper model name to use
            api_key: OpenAI API key for embeddings
        """
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.openai_embeddings = None
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            self.openai_embeddings = OpenAIEmbeddings()
        else:
            print("Warning: No API key provided. Embedding functionality will be limited.")
        
# Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(in_memory=True)
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "whisper_audio_collection"
        
        # Create collection
        qdrant_connector.create_whisper_collection(self.collection_name)
        
        # Set up hybrid search if embeddings available
        self.use_hybrid_search = False
        if self.openai_embeddings:
            try:
                from langchain_qdrant import RetrievalMode
                self.retrieval_mode = RetrievalMode.HYBRID
                self.use_hybrid_search = True
                print("Hybrid search capability enabled")
            except ImportError:
                print("Hybrid search not available - using standard search")
        
        # Initialize metrics loader
        self.wer_metric = load("wer")
        self.cer_metric = load("cer")

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and return as numpy array"""
        print(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=16000)
        return audio, sr
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper model"""
        print("Transcribing audio with Whisper...")
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        if self.device == "cuda":
            input_features = input_features.to("cuda")
        
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    
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
        chunks = []
        
        # Split on silence
        audio_chunks = []
        from pydub.silence import split_on_silence
        
        audio_chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        print(f"Created {len(audio_chunks)} audio chunks based on silence detection")
        return audio_chunks
    
    def chunk_text_fixed_size(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Chunk text into fixed-size segments"""
        print(f"Chunking text into fixed-size segments (size={chunk_size}, overlap={chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        documents = text_splitter.create_documents([text])
        print(f"Created {len(documents)} fixed-size text chunks")
        return documents
    
    def chunk_text_recursive(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Chunk text using RecursiveCharacterTextSplitter with optimal settings"""
        print(f"Chunking text using RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})...")
        
        # Create a text splitter optimized for audio transcription
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Use separators optimized for transcriptions
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        documents = text_splitter.create_documents([text])
        print(f"Created {len(documents)} recursive text chunks")
        return documents
    
    def chunk_text_by_sentences(self, text: str) -> List[Document]:
        """Chunk text by sentences"""
        print("Chunking text by sentences...")
        sentences = sent_tokenize(text)
        documents = [Document(page_content=sentence) for sentence in sentences]
        print(f"Created {len(documents)} sentence chunks")
        return documents
    
    def vectorize_and_store(self, documents: List[Document]) -> None:
        """Vectorize and store documents in Qdrant"""
        if not self.openai_embeddings:
            raise ValueError("OpenAI API key is required for vectorization")
        
        print("Vectorizing and storing documents in Qdrant...")
        
        try:
            # Standard vector storage (more reliable)
            from langchain_community.vectorstores import Qdrant
            
            Qdrant.from_documents(
                documents=documents,
                embedding=self.openai_embeddings,
                client=self.qdrant_client,
                collection_name=self.collection_name
            )
            print(f"Stored {len(documents)} documents in Qdrant collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error storing documents in Qdrant: {e}")
            # Save documents to disk as fallback
            output_dir = "chunks/whisper/fallback"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print(f"Saving documents to {output_dir} as fallback")
            for i, doc in enumerate(documents):
                with open(f"{output_dir}/chunk_{i}.txt", "w") as f:
                    f.write(doc.page_content)
                with open(f"{output_dir}/chunk_{i}_metadata.json", "w") as f:
                    json.dump(doc.metadata, f, indent=2)
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents based on query"""
        if not self.openai_embeddings:
            raise ValueError("OpenAI API key is required for search")
        
        print(f"Searching for: '{query}'")
        
        # Check if hybrid search is available
        if self.use_hybrid_search:
            try:
                from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
                
                # Setup sparse embedding for hybrid search
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                
                # Create vector store with hybrid capabilities
                vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.openai_embeddings,
                    sparse_embedding=sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID
                )
                print("Using hybrid search (dense + sparse vectors)")
            except Exception as e:
                print(f"Failed to use hybrid search, falling back to standard: {e}")
                self.use_hybrid_search = False
                vector_store = self.create_standard_vector_store()
        else:
            # Standard vector storage
            vector_store = self.create_standard_vector_store()
        
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        return results
        
    def create_standard_vector_store(self):
        """Create standard vector store with OpenAI embeddings"""
        from langchain_community.vectorstores import Qdrant
        
        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.openai_embeddings,
        )
    
    def evaluate_transcription(self, reference_text: str, hypothesis_text: str) -> Dict[str, float]:
        """Evaluate transcription quality"""
        wer = self.wer_metric.compute(predictions=[hypothesis_text], references=[reference_text])
        cer = self.cer_metric.compute(predictions=[hypothesis_text], references=[reference_text])
        
        return {
            "wer": wer,
            "cer": cer
        }
    
    def evaluate_search_quality(self, query: str, expected_results: List[str], actual_results: List[Document]) -> Dict[str, float]:
        """Evaluate search quality"""
        actual_texts = [doc.page_content for doc in actual_results]
        
        # Precision: How many of the results are relevant
        precision = len(set(actual_texts) & set(expected_results)) / len(actual_texts) if actual_texts else 0
        
        # Recall: How many of the relevant documents were retrieved
        recall = len(set(actual_texts) & set(expected_results)) / len(expected_results) if expected_results else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def process_audio(self, file_path: str, chunking_method: str = "fixed_text", chunk_param: int = 1000) -> Dict:
        """
        Process audio end-to-end: load, transcribe, chunk, vectorize, store
        
        Args:
            file_path: Path to audio file
            chunking_method: Method to use for chunking ('fixed_audio', 'silence', 'fixed_text', 'semantic', 'sentences')
            chunk_param: Parameter to use for chunking (duration in seconds for audio, chunk size for text)
            
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
        
        # Transcribe whole audio
        full_transcription = self.transcribe_audio(audio)
        results["full_transcription"] = full_transcription
        
        # Chunk based on method
        if chunking_method == "fixed_audio":
            # Chunk audio, then transcribe each chunk
            audio_chunks = self.chunk_audio_fixed_size(audio, sr, chunk_param)
            documents = []
            
            for i, chunk in enumerate(audio_chunks):
                chunk_text = self.transcribe_audio(chunk)
                documents.append(Document(page_content=chunk_text, metadata={"chunk_id": i}))
            
            results["chunk_count"] = len(documents)
            results["documents"] = documents
            
        elif chunking_method == "silence":
            # Chunk audio by silence, then transcribe
            audio_chunks = self.chunk_audio_by_silence(file_path, min_silence_len=chunk_param)
            documents = []
            
            for i, chunk in enumerate(audio_chunks):
                # Save chunk temporarily
                temp_path = f"temp_chunk_{i}.wav"
                chunk.export(temp_path, format="wav")
                
                # Load and transcribe
                chunk_audio, chunk_sr = self.load_audio(temp_path)
                chunk_text = self.transcribe_audio(chunk_audio)
                documents.append(Document(page_content=chunk_text, metadata={"chunk_id": i}))
                
                # Clean up temp file
                os.remove(temp_path)
            
            results["chunk_count"] = len(documents)
            results["documents"] = documents
            
        elif chunking_method == "fixed_text":
            # Transcribe, then chunk the text
            documents = self.chunk_text_fixed_size(full_transcription, chunk_size=chunk_param)
            results["chunk_count"] = len(documents)
            results["documents"] = documents
            
        elif chunking_method == "recursive":
            # Transcribe, then chunk recursively with optimized settings
            documents = self.chunk_text_recursive(full_transcription, chunk_size=chunk_param)
            results["chunk_count"] = len(documents)
            results["documents"] = documents
            
        elif chunking_method == "sentences":
            # Transcribe, then chunk by sentences
            documents = self.chunk_text_by_sentences(full_transcription)
            results["chunk_count"] = len(documents)
            results["documents"] = documents
            
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
        
        # Vectorize and store
        self.vectorize_and_store(documents)
        
        # Calculate processing time
        end_time = time.time()
        results["processing_time"] = end_time - start_time
        
        return results

    def run_benchmark(self, file_path: str, reference_text: str, test_queries: List[Dict[str, Union[str, List[str]]]]):
        """
        Run full benchmark on audio file with different chunking methods
        
        Args:
            file_path: Path to audio file
            reference_text: Reference transcription for evaluation
            test_queries: List of dicts with 'query' and 'expected_results' keys
        """
        chunking_methods = [
            {"name": "fixed_audio", "param": 30},  # 30 second chunks
            {"name": "silence", "param": 500},     # 500ms silence detection
            {"name": "fixed_text", "param": 1000}, # 1000 char chunks
            {"name": "sentences", "param": 0},     # Sentence chunking
            {"name": "recursive", "param": 500},   # Recursive chunking with 500 char size
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
            
            # Evaluate transcription
            transcription_metrics = self.evaluate_transcription(
                reference_text=reference_text,
                hypothesis_text=processing_results["full_transcription"]
            )
            
            # Evaluate search for each query
            search_metrics = []
            for test_query in test_queries:
                query = test_query["query"]
                expected = test_query["expected_results"]
                
                # Perform search
                search_results = self.search(query, k=3)
                
                # Evaluate results
                query_metrics = self.evaluate_search_quality(
                    query=query,
                    expected_results=expected,
                    actual_results=search_results
                )
                
                search_metrics.append({
                    "query": query,
                    "metrics": query_metrics,
                    "search_results": [r.page_content for r in search_results]
                })
            
            # Aggregate results
            result = {
                "chunking_method": method["name"],
                "chunk_param": method["param"],
                "transcription_metrics": transcription_metrics,
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
                    size=1536,
                    distance=models.Distance.COSINE,
                ),
            )
        
        # Display and return results
        self.display_benchmark_results(results)
        return results
    
    def display_benchmark_results(self, results: List[Dict]) -> None:
        """Display benchmark results in a structured format"""
        print("\n" + "="*80)
        print("WHISPER APPROACH BENCHMARK RESULTS")
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
                "WER": round(result["transcription_metrics"]["wer"], 4),
                "CER": round(result["transcription_metrics"]["cer"], 4),
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
                print(f"  Query: '{query_result['query']}'")
                print(f"  Precision: {round(query_result['metrics']['precision'], 4)}")
                print(f"  Recall: {round(query_result['metrics']['recall'], 4)}")
                print(f"  F1 Score: {round(query_result['metrics']['f1_score'], 4)}")
                print(f"  Top Results: {query_result['search_results'][:2]}")
                print()

    def test_search(self, query_text: str):
        """Test search with a specific query"""
        print(f"\n===== WHISPER TEXT SEARCH =====")
        print(f"Query: '{query_text}'")
        
        try:
            results = self.search(query_text, k=2)
            print(f"\nResults:")
            for i, doc in enumerate(results):
                print(f"  Result {i+1}: {doc.page_content}")
            return results
        except Exception as e:
            print(f"  Search failed: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key here
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    
    # Initialize processor - try with API key if available
    processor = WhisperAudioProcessor(api_key=OPENAI_API_KEY)
    
    # Use test audio file
    audio_file = "test_audio.wav"
    
    # Run simple demo or a modified benchmark without vectorization?
    RUN_BENCHMARK = False
    # Note: If RUN_BENCHMARK is True but no OpenAI API key is available,
    # we'll run a simplified benchmark that skips the vectorization step
    
    if RUN_BENCHMARK:
        print("\n===== RUNNING WHISPER BENCHMARK =====")
        # Reference text (based on our transcription)
        reference_text = "Wow, what an audience, but if I'm being honest, I don't care what you think of my talk."
        
        if not OPENAI_API_KEY:
            print("No OpenAI API key available, running simplified chunking benchmark (no vectorization)")
            
            # Simplified benchmark without vectorization
            chunking_methods = [
                {"name": "fixed_text", "param": 100},   # Small chunks
                {"name": "fixed_text", "param": 500},   # Medium chunks
                {"name": "fixed_text", "param": 1000},  # Large chunks
                {"name": "recursive", "param": 500}     # Recursive with 500 chars
            ]
            
            # Load and transcribe audio
            print("\nLoading and transcribing audio...")
            audio, sr = processor.load_audio(audio_file)
            transcription = processor.transcribe_audio(audio)
            
            # Evaluate transcription quality using WER
            wer_score = processor.evaluate_transcription(reference_text, transcription[:len(reference_text)])
            print(f"\nTranscription Quality WER: {wer_score['wer']:.4f}")
            print(f"Transcription Quality CER: {wer_score['cer']:.4f}")
            
            print("\nBenchmarking different chunking methods:")
            
            # Test different chunking methods
            results = []
            for method in chunking_methods:
                start_time = time.time()
                
                if method["name"] == "fixed_text":
                    chunks = processor.chunk_text_fixed_size(transcription, chunk_size=method["param"])
                    method_name = f"Fixed ({method['param']} chars)"
                elif method["name"] == "sentences":
                    chunks = processor.chunk_text_by_sentences(transcription)
                    method_name = "Sentences"
                elif method["name"] == "recursive":
                    chunks = processor.chunk_text_recursive(transcription, chunk_size=method["param"])
                    method_name = f"Recursive ({method['param']} chars)"
                else:
                    continue  # Skip unsupported methods
                
                end_time = time.time()
                
                results.append({
                    "method": method_name, 
                    "chunks": len(chunks), 
                    "time": end_time - start_time,
                    "avg_length": sum(len(doc.page_content) for doc in chunks) / len(chunks) if chunks else 0
                })
            
            # Print results as a table
            print("\nChunking Results:")
            print(f"{'Method':<25} {'Chunks':<10} {'Avg Length':<15} {'Time (s)':<10}")
            print("-" * 60)
            for result in results:
                print(f"{result['method']:<25} {result['chunks']:<10} {result['avg_length']:<15.1f} {result['time']:<10.3f}")
        else:
            print("OpenAI API key is available, running full benchmark including vectorization")
            # Test queries for the benchmark
            test_queries = [
                {
                    "query": "What does the speaker say about the audience?",
                    "expected_results": ["Wow, what an audience"]
                },
                {
                    "query": "Why doesn't the speaker care about the audience?",
                    "expected_results": ["I don't care what you think of my talk"]
                }
            ]
            
            # Run benchmark with various chunking methods
            benchmark_results = processor.run_benchmark(
                file_path=audio_file,
                reference_text=reference_text,
                test_queries=test_queries
            )
    else:
        # Basic audio processing with enhanced chunking and vectorization
        print("\n===== AUDIO TRANSCRIPTION =====")
        print("Loading and transcribing audio...")
        audio, sr = processor.load_audio(audio_file)
        transcription = processor.transcribe_audio(audio)
        print("\nTranscription:")
        print(transcription[:500] + "..." if len(transcription) > 500 else transcription)
        
        print("\n===== TEXT CHUNKING =====")
        # Create chunks with recursive text splitter - optimized for audio transcription
        print("\nCreating recursive text chunks...")
        chunk_size = 500  # Smaller chunks for better retrieval
        documents = processor.chunk_text_recursive(transcription, chunk_size=chunk_size)
        print(f"Created {len(documents)} chunks")
        
        # Print first few chunks
        print("\nSample chunks:")
        for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
            print(f"Chunk {i}: {doc.page_content}")
        
        # If OpenAI API key is available, store in Qdrant and test search
        if OPENAI_API_KEY:
            print("\n===== VECTOR STORAGE AND SEARCH =====")
            # Store in vector database
            processor.vectorize_and_store(documents)
            
            # Test search functionality
            test_query = "What is this audio about?"
            print(f"\nSearching for: '{test_query}'")
            results = processor.search(test_query, k=2)
            
            # Display results
            print(f"\nTop {len(results)} results:")
            for i, doc in enumerate(results):
                print(f"Result {i+1}: {doc.page_content}")
        
        print("\n===== WHISPER PROCESSING COMPLETED SUCCESSFULLY! =====")

    # Run test search if OpenAI API key is available
    if OPENAI_API_KEY:
        processor.test_search("What does the speaker say about the audience?")
