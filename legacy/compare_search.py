import os
import numpy as np
from whisper import WhisperAudioProcessor
from wave2vec import Wav2VecProcessor
from clap import ClapProcessor

def main():
    # Text query to use across all methods that support text search
    text_query = "What does the speaker say about the audience?"
    
    # Audio file to use for queries and processing
    audio_file = "test_audio.wav"
    
    print("\n" + "="*80)
    print("COMPARING SEARCH METHODS USING THE SAME QUERIES")
    print("="*80)
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("\nWARNING: No OpenAI API key found. Whisper search will not work properly.")
        print("Set your OpenAI API key as an environment variable to enable Whisper search.")
    
    # =============== PROCESSING ===============
    # Initialize processors
    print("\nInitializing processors...")
    whisper_processor = WhisperAudioProcessor(api_key=openai_api_key)
    wav2vec_processor = Wav2VecProcessor()
    clap_processor = ClapProcessor()
    
    # Process audio with each method
    print("\nProcessing audio with Whisper...")
    if openai_api_key:
        whisper_processor.process_audio(audio_file, chunking_method="recursive", chunk_param=500)
    else:
        print("  Skipping Whisper processing (no API key)")
    
    print("\nProcessing audio with Wav2Vec...")
    wav2vec_processor.process_audio(audio_file, chunking_method="fixed", chunk_param=5)
    
    print("\nProcessing audio with CLAP...")
    clap_processor.process_audio(audio_file, chunking_method="fixed", chunk_param=5)
    
    # =============== SEARCH TESTING ===============
    # Create audio query (5-second segment starting at 10s)
    audio, sr = wav2vec_processor.load_audio(audio_file)
    audio_segment = audio[10*sr:15*sr]  # 5-second segment from the audio
    
    # Run text search tests
    print("\n" + "="*80)
    print("TEXT SEARCH COMPARISON")
    print("="*80)
    
    # Test Whisper text search
    if openai_api_key:
        whisper_processor.test_search(text_query)
    else:
        print("\nWHISPER TEXT SEARCH: Skipped (no API key)")
    
    # Test CLAP text search
    clap_processor.test_search_by_text(text_query)
    
    # Run audio search tests
    print("\n" + "="*80)
    print("AUDIO SEARCH COMPARISON")
    print("="*80)
    
    # Test Wav2Vec audio search
    wav2vec_processor.test_search(audio_segment)
    
    # Test CLAP audio search
    clap_processor.test_search_by_audio(audio_segment)
    
    print("\n" + "="*80)
    print("SEARCH COMPARISON COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 