import os
import numpy as np
import argparse
from whisper import WhisperAudioProcessor
from wave2vec import Wav2VecProcessor
from clap import ClapProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Compare different audio vectorization and search methods')
    parser.add_argument('--audio_file', type=str, default="test_audio.wav", 
                        help='Audio file to process')
    parser.add_argument('--text_query', type=str, default="Who has a foreign trade zone manual?", 
                        help='Text query to search for')
    parser.add_argument('--chunking_method', type=str, default="recursive", 
                        choices=["fixed", "recursive", "sentences"], 
                        help='Chunking method to use')
    parser.add_argument('--chunk_param', type=int, default=500, 
                        help='Parameter for chunking (e.g., chunk size for text, seconds for audio)')
    parser.add_argument('--skip_whisper', action='store_true', 
                        help='Skip Whisper processing')
    parser.add_argument('--skip_wav2vec', action='store_true', 
                        help='Skip Wav2Vec processing')
    parser.add_argument('--skip_clap', action='store_true', 
                        help='Skip CLAP processing')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Text query to use across all methods that support text search
    text_query = args.text_query
    
    # Audio file to use for queries and processing
    audio_file = args.audio_file
    
    print("\n" + "="*80)
    print("COMPARING SEARCH METHODS USING THE SAME QUERIES")
    print("="*80)
    print(f"Audio file: {audio_file}")
    print(f"Text query: {text_query}")
    print(f"Chunking method: {args.chunking_method}")
    print(f"Chunk parameter: {args.chunk_param}")
    print("="*80)
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("\nWARNING: No OpenAI API key found. Whisper search will not work properly.")
        print("Set your OpenAI API key as an environment variable to enable Whisper search.")
    
    # =============== PROCESSING ===============
    # Initialize processors
    print("\nInitializing processors...")
    
    processors = []
    
    if not args.skip_whisper:
        whisper_processor = WhisperAudioProcessor(api_key=openai_api_key)
        processors.append(("Whisper", whisper_processor, args.chunking_method, args.chunk_param))
    
    if not args.skip_wav2vec:
        wav2vec_processor = Wav2VecProcessor()
        # For wav2vec, we use fixed chunking with a duration in seconds
        processors.append(("Wav2Vec", wav2vec_processor, "fixed", 5))
    
    if not args.skip_clap:
        clap_processor = ClapProcessor()
        # For CLAP, we use fixed chunking with a duration in seconds
        processors.append(("CLAP", clap_processor, "fixed", 5))
    
    # Process audio with each method
    for name, processor, method, param in processors:
        print(f"\nProcessing audio with {name}...")
        
        try:
            if name == "Whisper":
                # For Whisper, use the specified method
                if openai_api_key:
                    processor.process_audio(audio_file, chunking_method=method, chunk_param=param)
                else:
                    print(f"  Skipping {name} processing (no API key)")
            else:
                # For audio-based methods, use fixed chunking
                processor.process_audio(audio_file, chunking_method=method, chunk_param=param)
            
            print(f"  {name} processing completed successfully")
        except Exception as e:
            print(f"  Error processing with {name}: {e}")
    
    # =============== SEARCH TESTING ===============
    # Create audio query (5-second segment starting at 10s)
    if not args.skip_wav2vec:
        audio, sr = wav2vec_processor.load_audio(audio_file)
        # Make sure we don't go out of bounds for short audio files
        start_sec = min(10, len(audio) // sr - 5)
        audio_segment = audio[start_sec*sr:(start_sec+5)*sr]  # 5-second segment from the audio
    
    # Run text search tests
    print("\n" + "="*80)
    print("TEXT SEARCH COMPARISON")
    print("="*80)
    
    # Test Whisper text search
    if not args.skip_whisper and openai_api_key:
        try:
            print("\nWHISPER TEXT SEARCH:")
            whisper_processor.test_search(text_query)
        except Exception as e:
            print(f"  Error in Whisper text search: {e}")
    else:
        print("\nWHISPER TEXT SEARCH: Skipped")
    
    # Test CLAP text search
    if not args.skip_clap:
        try:
            print("\nCLAP TEXT SEARCH:")
            clap_processor.test_search_by_text(text_query)
        except Exception as e:
            print(f"  Error in CLAP text search: {e}")
    else:
        print("\nCLAP TEXT SEARCH: Skipped")
    
    # Run audio search tests
    print("\n" + "="*80)
    print("AUDIO SEARCH COMPARISON")
    print("="*80)
    
    # Test Wav2Vec audio search
    if not args.skip_wav2vec:
        try:
            print("\nWAV2VEC AUDIO SEARCH:")
            wav2vec_processor.test_search(audio_segment)
        except Exception as e:
            print(f"  Error in Wav2Vec audio search: {e}")
    else:
        print("\nWAV2VEC AUDIO SEARCH: Skipped")
    
    # Test CLAP audio search
    if not args.skip_clap and not args.skip_wav2vec:  # Need Wav2Vec to get audio segment
        try:
            print("\nCLAP AUDIO SEARCH:")
            clap_processor.test_search_by_audio(audio_segment)
        except Exception as e:
            print(f"  Error in CLAP audio search: {e}")
    else:
        print("\nCLAP AUDIO SEARCH: Skipped")
    
    print("\n" + "="*80)
    print("SEARCH COMPARISON COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main() 