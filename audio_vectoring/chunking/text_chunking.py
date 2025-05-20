from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import nltk
from nltk.tokenize import sent_tokenize

# Make sure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def chunk_text_fixed_size(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Chunk text into fixed-size segments
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    print(f"Chunking text into fixed-size segments (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} fixed-size text chunks")
    return chunks

def chunk_text_recursive(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Chunk text using RecursiveCharacterTextSplitter with optimal settings for audio transcription
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    print(f"Chunking text using RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})...")
    
    # Create a text splitter optimized for audio transcription
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Use separators optimized for transcriptions
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} recursive text chunks")
    return chunks

def chunk_text_by_sentences(text: str, max_words: int = 50) -> List[str]:
    """
    Chunk text by sentences using NLTK's sentence tokenizer
    
    Args:
        text: Text to chunk
        max_words: Maximum words per chunk (will combine short sentences)
        
    Returns:
        List of sentence chunks
    """
    print("Chunking text by sentences...")
    sentences = sent_tokenize(text)
    
    # Combine short sentences into larger chunks based on word count
    chunks = []
    current_chunk = ""
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        if current_word_count + sentence_word_count <= max_words:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_word_count += sentence_word_count
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            current_word_count = sentence_word_count
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} sentence chunks")
    return chunks 