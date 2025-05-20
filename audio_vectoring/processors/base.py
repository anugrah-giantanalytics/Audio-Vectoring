import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain_core.documents import Document

class BaseAudioProcessor(ABC):
    """
    Base class for audio processors.
    All audio processors should inherit from this class and implement its methods.
    """
    
    @abstractmethod
    def process_audio(self, audio_file: str, chunking_method: str, chunk_param: int, save_results: bool = True) -> List[Document]:
        """
        Process an audio file: load, convert to desired format, chunk, embed, and store.
        
        Args:
            audio_file: Path to the audio file
            chunking_method: Method to use for chunking
            chunk_param: Parameter for chunking (size, overlap, etc.)
            save_results: Whether to save results to disk
            
        Returns:
            List of processed document chunks
        """
        pass
    
    @abstractmethod
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file and return as numpy array.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass
    
    @abstractmethod
    def test_search(self, query: Union[str, np.ndarray]) -> List[Any]:
        """
        Test search functionality with a query.
        
        Args:
            query: Text query or audio segment to search with
            
        Returns:
            List of search results
        """
        pass 