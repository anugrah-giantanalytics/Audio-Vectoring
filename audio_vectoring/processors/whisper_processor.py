import os
import json
import numpy as np
import torch
import time
from typing import List, Dict, Any, Optional, Union, Tuple

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from evaluate import load

from audio_vectoring.processors.base import BaseAudioProcessor
from audio_vectoring.utils.audio_utils import load_audio, ensure_dir
from audio_vectoring.chunking.text_chunking import (
    chunk_text_fixed_size,
    chunk_text_recursive,
    chunk_text_by_sentences
)
from audio_vectoring.storage.qdrant_connector import QdrantConnector

class WhisperAudioProcessor(BaseAudioProcessor):
    """
    Audio processor using OpenAI's Whisper model for transcription,
    then creating text embeddings from the transcription.
    """
    
    def __init__(self, 
                 model_name: str = "openai/whisper-base", 
                 api_key: Optional[str] = None,
                 collection_name: str = "whisper_embeddings",
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None,
                 in_memory: bool = True):
        """
        Initialize WhisperAudioProcessor.
        
        Args:
            model_name: Whisper model name/size to use
            api_key: OpenAI API key for embeddings
            collection_name: Name of the Qdrant collection to use
            qdrant_url: URL for Qdrant server (optional)
            qdrant_api_key: API key for Qdrant Cloud (optional)
            in_memory: Whether to use in-memory Qdrant database
        """
        super().__init__()
        
        # Initialize Whisper model
        self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Initialize embedding model
        self.openai_api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for WhisperAudioProcessor")
        
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        # Initialize Qdrant vector store
        vector_size = 1536  # Size of OpenAI's text-embedding-3-small model
        self.vector_db = QdrantConnector(
            collection_name=collection_name,
            vector_size=vector_size,
            url=qdrant_url,
            api_key=qdrant_api_key,
            in_memory=in_memory
        )
        
        # Initialize metrics
        self.wer = load("wer")
        self.current_transcript = None
        self.current_audio_file = None
        
        # Track processed chunks
        self.chunks = []
        self.processing_time = 0
        
    def process_audio(self, audio_file: str, chunking_method: str = "fixed", 
                     chunk_param: int = 500, save_results: bool = True) -> List[Document]:
        """
        Process audio file: transcribe, chunk text, embed, and store in vector DB.
        
        Args:
            audio_file: Path to audio file
            chunking_method: Method for chunking text ('fixed', 'recursive', 'sentence')
            chunk_param: Parameter for chunking (chunk size or overlap)
            save_results: Whether to save results to disk
            
        Returns:
            List of processed document chunks
        """
        print(f"Processing audio file: {audio_file}")
        start_time = time.time()
        
        # Load and transcribe audio
        self.current_audio_file = audio_file
        self.current_transcript = self.transcribe_audio(audio_file)
        
        print(f"Transcription completed. Length: {len(self.current_transcript)}")
        
        # Chunk text based on selected method
        if chunking_method == "fixed":
            chunks = chunk_text_fixed_size(self.current_transcript, chunk_size=chunk_param)
        elif chunking_method == "recursive":
            chunks = chunk_text_recursive(self.current_transcript, chunk_size=chunk_param)
        elif chunking_method == "sentence":
            chunks = chunk_text_by_sentences(self.current_transcript, max_words=chunk_param)
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
        
        # Convert chunks to Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": os.path.basename(audio_file),
                    "chunk_id": i,
                    "chunking_method": chunking_method,
                    "processor": "whisper",
                }
            )
            documents.append(doc)
        
        # Embed and store chunks
        self._embed_and_store(documents)
        
        # Save results if requested
        if save_results:
            self._save_results(documents, audio_file, chunking_method)
        
        self.chunks = documents
        self.processing_time = time.time() - start_time
        
        print(f"Audio processing completed in {self.processing_time:.2f} seconds")
        print(f"Created {len(documents)} text chunks")
        
        return documents
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file and return as numpy array.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        return load_audio(file_path)
    
    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe audio file using Whisper model.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Load audio
        audio, sr = self.load_audio(audio_file)
        
        # Process through Whisper model
        input_features = self.whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        
        # Generate token ids
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features)
        
        # Decode token ids to text
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    def _embed_and_store(self, documents: List[Document]):
        """
        Embed document chunks and store in Qdrant.
        
        Args:
            documents: List of Document objects to embed and store
        """
        texts = [doc.page_content for doc in documents]
        
        # Create embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Store documents and embeddings in Qdrant
        self.vector_db.add_documents(documents, embeddings)
        
        print(f"Embedded and stored {len(documents)} documents in Qdrant")
    
    def _save_results(self, documents: List[Document], audio_file: str, chunking_method: str):
        """
        Save processing results to disk.
        
        Args:
            documents: Processed document chunks
            audio_file: Source audio file path
            chunking_method: Chunking method used
        """
        base_dir = os.path.join("chunk_results", "whisper")
        ensure_dir(base_dir)
        
        # Save full transcript
        transcript_path = os.path.join(base_dir, f"{os.path.basename(audio_file)}_transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(self.current_transcript)
        
        # Save individual chunks
        for i, doc in enumerate(documents):
            chunk_path = os.path.join(base_dir, f"chunk_{i}.txt")
            with open(chunk_path, "w") as f:
                f.write(doc.page_content)
            
            # Save metadata
            metadata_path = os.path.join(base_dir, f"chunk_{i}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(doc.metadata, f, indent=2)
        
        # Save processing info
        info_path = os.path.join(base_dir, "processing_info.json")
        info = {
            "audio_file": audio_file,
            "chunking_method": chunking_method,
            "num_chunks": len(documents),
            "processing_time": self.processing_time,
            "whisper_model": self.whisper_model.config._name_or_path,
            "embedding_model": "text-embedding-3-small",
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
            
        print(f"Saved results to {base_dir}")
    
    def test_search(self, query: Union[str, np.ndarray]) -> List[Document]:
        """
        Test search functionality using a text query or audio segment.
        
        Args:
            query: Text query or audio segment
            
        Returns:
            List of Document objects matching the query
        """
        if not self.chunks:
            raise ValueError("No data has been processed yet. Call process_audio first.")
        
        # Handle different query types
        if isinstance(query, str):
            # Text query - create embedding directly
            query_embedding = self.embedding_model.embed_query(query)
        else:
            # Audio segment - transcribe first, then embed
            transcription = self.transcribe_audio_segment(query)
            query_embedding = self.embedding_model.embed_query(transcription)
        
        # Search in vector DB
        results = self.vector_db.search(query_embedding, limit=3)
        
        # Extract documents from results
        documents = [res["document"] for res in results]
        
        return documents
    
    def transcribe_audio_segment(self, audio_segment: np.ndarray, sr: int = 16000) -> str:
        """
        Transcribe an audio segment (not a file).
        
        Args:
            audio_segment: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Transcribed text
        """
        # Process through Whisper model
        input_features = self.whisper_processor(audio_segment, sampling_rate=sr, return_tensors="pt").input_features
        
        # Generate token ids
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features)
        
        # Decode token ids to text
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
        
    def evaluate_transcription(self, reference_text: str) -> float:
        """
        Evaluate transcription quality against reference text.
        
        Args:
            reference_text: Ground truth text for comparison
            
        Returns:
            Word Error Rate (WER) score
        """
        if not self.current_transcript:
            raise ValueError("No transcription available. Process an audio file first.")
        
        wer_score = self.wer.compute(predictions=[self.current_transcript], 
                                     references=[reference_text])
        
        return wer_score 