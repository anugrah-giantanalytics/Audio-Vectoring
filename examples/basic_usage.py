#!/usr/bin/env python3
"""
Basic usage examples for audio vectorization processors.

This script demonstrates how to use each of the three audio processors:
1. Whisper (text-based)
2. Wav2Vec (direct audio)
3. CLAP (multimodal)
"""

import os
import numpy as np
from audio_vectoring.processors.whisper_processor import WhisperAudioProcessor
from audio_vectoring.processors.wav2vec_processor import Wav2VecProcessor
from audio_vectoring.processors.clap_processor import ClapProcessor

def example_whisper():
    """Example using the Whisper processor"""
    print("\n" + "="*50)
    print("WHISPER PROCESSOR EXAMPLE (Text-based)")
    print("="*50)
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Skipping Whisper example.")
        print("Set the OPENAI_API_KEY environment variable to run this example.")
        return
    
    # Initialize processor
    processor = WhisperAudioProcessor(api_key=api_key)
    
    # Process audio file
    audio_file = "test_audio.wav"
    print(f"\nProcessing audio file: {audio_file}")
    results = processor.process_audio(
        file_path=audio_file,
        chunking_method="recursive",
        chunk_param=500
    )
    
    # Display results
    print("\nProcessing results:")
    print(f"  Audio duration: {results['audio_duration']:.2f} seconds")
    print(f"  Chunk count: {results['chunk_count']}")
    print(f"  Processing time: {results['processing_time']:.2f} seconds")
    
    # Run a search
    query = "what does the speaker say about the audience"
    print(f"\nSearching for: '{query}'")
    search_results = processor.test_search(query)
    
    # Display search results
    if search_results:
        print("\nTop result content:")
        print(f"  {search_results[0].page_content}")

def example_wav2vec():
    """Example using the Wav2Vec processor"""
    print("\n" + "="*50)
    print("WAV2VEC PROCESSOR EXAMPLE (Direct Audio)")
    print("="*50)
    
    # Initialize processor
    processor = Wav2VecProcessor()
    
    # Process audio file
    audio_file = "test_audio.wav"
    print(f"\nProcessing audio file: {audio_file}")
    results = processor.process_audio(
        file_path=audio_file,
        chunking_method="fixed",
        chunk_param=5  # 5-second chunks
    )
    
    # Display results
    print("\nProcessing results:")
    print(f"  Audio duration: {results['audio_duration']:.2f} seconds")
    print(f"  Chunk count: {results['chunk_count']}")
    print(f"  Processing time: {results['processing_time']:.2f} seconds")
    
    # Create a query from a segment of the audio
    print("\nCreating audio query from 10-15 second segment")
    audio, sr = processor.load_audio(audio_file)
    query_segment = audio[10*sr:15*sr]  # 5-second segment from 10s to 15s
    
    # Run a search
    print("Searching for similar audio segments")
    search_results = processor.test_search(query_segment)
    
    # Display search results
    if search_results:
        print("\nTop result metadata:")
        print(f"  Chunk ID: {search_results[0]['metadata']['chunk_id']}")
        print(f"  Score: {search_results[0]['score']:.4f}")
        if "start_time" in search_results[0]["metadata"]:
            print(f"  Time range: {search_results[0]['metadata']['start_time']:.2f}s - "
                  f"{search_results[0]['metadata']['end_time']:.2f}s")

def example_clap():
    """Example using the CLAP processor"""
    print("\n" + "="*50)
    print("CLAP PROCESSOR EXAMPLE (Multimodal)")
    print("="*50)
    
    # Initialize processor
    processor = ClapProcessor()
    
    # Process audio file
    audio_file = "test_audio.wav"
    print(f"\nProcessing audio file: {audio_file}")
    results = processor.process_audio(
        file_path=audio_file,
        chunking_method="fixed",
        chunk_param=5  # 5-second chunks
    )
    
    # Display results
    print("\nProcessing results:")
    print(f"  Audio duration: {results['audio_duration']:.2f} seconds")
    print(f"  Chunk count: {results['chunk_count']}")
    print(f"  Processing time: {results['processing_time']:.2f} seconds")
    
    # Run a text-to-audio search
    text_query = "what does the speaker say about the audience"
    print(f"\nText-to-audio search for: '{text_query}'")
    text_results = processor.test_search_by_text(text_query)
    
    # Display text search results
    if text_results:
        print("\nTop text search result:")
        print(f"  Chunk ID: {text_results[0]['metadata']['chunk_id']}")
        print(f"  Score: {text_results[0]['score']:.4f}")
        if "start_time" in text_results[0]["metadata"]:
            print(f"  Time range: {text_results[0]['metadata']['start_time']:.2f}s - "
                  f"{text_results[0]['metadata']['end_time']:.2f}s")
    
    # Create an audio query
    print("\nCreating audio query from 10-15 second segment")
    audio, sr = processor.load_audio(audio_file)
    query_segment = audio[10*sr:15*sr]  # 5-second segment from 10s to 15s
    
    # Run an audio-to-audio search
    print("Audio-to-audio search")
    audio_results = processor.test_search_by_audio(query_segment)
    
    # Display audio search results
    if audio_results:
        print("\nTop audio search result:")
        print(f"  Chunk ID: {audio_results[0]['metadata']['chunk_id']}")
        print(f"  Score: {audio_results[0]['score']:.4f}")
        if "start_time" in audio_results[0]["metadata"]:
            print(f"  Time range: {audio_results[0]['metadata']['start_time']:.2f}s - "
                  f"{audio_results[0]['metadata']['end_time']:.2f}s")

def main():
    """Run all examples"""
    print("AUDIO VECTORIZATION EXAMPLES")
    
    # Check if test audio exists
    if not os.path.exists("test_audio.wav"):
        print("\nError: test_audio.wav not found.")
        print("Please place a test audio file named 'test_audio.wav' in the current directory.")
        return
    
    # Run examples for each processor
    example_whisper()
    example_wav2vec()
    example_clap()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 