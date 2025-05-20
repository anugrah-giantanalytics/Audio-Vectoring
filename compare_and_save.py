import os
import numpy as np
import json
import librosa
from whisper import WhisperAudioProcessor
from wave2vec import Wav2VecProcessor
from clap import ClapProcessor
from langchain.docstore.document import Document

def ensure_dir(directory):
    """Make sure the directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_text_chunk(chunk, file_path):
    """Save text chunk to a file"""
    with open(file_path, 'w') as f:
        f.write(chunk.page_content)
        
    # Save metadata in a separate file
    metadata_path = file_path.replace('.txt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(chunk.metadata, f, indent=2)
    
    return file_path

def save_audio_chunk(result, audio_file, output_dir, index, method):
    """Save audio chunk and its metadata"""
    # Extract metadata
    chunk_id = result['metadata']['chunk_id']
    score = result['score']
    
    # Create filename
    base_filename = f"{method}_result_{index+1}_chunk_{chunk_id}"
    metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
    
    # Save metadata
    metadata = {
        "score": float(score),
        "chunk_id": chunk_id,
        "rank": index + 1
    }
    
    # Add any additional metadata available
    for key, value in result['metadata'].items():
        if key != 'chunk_id':  # Already added
            metadata[key] = value
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def main():
    # Create base results directory
    results_dir = "chunk_results"
    ensure_dir(results_dir)
    
    # Audio file to use for search
    audio_file = "test_audio.wav"
    
    # Common query to use for text-based searches
    text_query = "what does the speaker say about the audience"
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("\nWARNING: No OpenAI API key found. Whisper search will not work properly.")
        print("Set your OpenAI API key as an environment variable to enable Whisper search.")
    
    print("\n" + "="*80)
    print("RUNNING COMPARISON SEARCH AND SAVING RESULTS")
    print("="*80)
    
    # =============== INITIALIZE PROCESSORS ===============
    print("\nInitializing processors...")
    whisper_processor = WhisperAudioProcessor(api_key=openai_api_key)
    wav2vec_processor = Wav2VecProcessor()
    clap_processor = ClapProcessor()
    
    # =============== PROCESS AUDIO ===============
    print("\nProcessing audio with each method...")

    # Process with Whisper
    if openai_api_key:
        print("\nProcessing audio with Whisper...")
        whisper_processor.process_audio(audio_file, chunking_method="recursive", chunk_param=500)
    else:
        print("\nSkipping Whisper processing (no API key found)")
    
    # Process with Wav2Vec
    print("\nProcessing audio with Wav2Vec...")
    wav2vec_processor.process_audio(audio_file, chunking_method="fixed", chunk_param=5)
    
    # Process with CLAP
    print("\nProcessing audio with CLAP...")
    clap_processor.process_audio(audio_file, chunking_method="fixed", chunk_param=5)
    
    # =============== CREATE AUDIO QUERY SEGMENT ===============
    # Create audio segment for audio-based search (5-second segment starting at 10s)
    audio, sr = wav2vec_processor.load_audio(audio_file)
    audio_segment = audio[10*sr:15*sr]  # 5-second segment
    
    # =============== RUN SEARCHES AND SAVE RESULTS ===============
    # Create result directories for each method
    whisper_dir = os.path.join(results_dir, "whisper")
    wav2vec_dir = os.path.join(results_dir, "wav2vec")
    clap_text_dir = os.path.join(results_dir, "clap_text")
    clap_audio_dir = os.path.join(results_dir, "clap_audio")
    
    ensure_dir(whisper_dir)
    ensure_dir(wav2vec_dir)
    ensure_dir(clap_text_dir)
    ensure_dir(clap_audio_dir)
    
    print("\n" + "="*80)
    print(f"SEARCH QUERY: '{text_query}'")
    print("="*80)
    
    # Run and save Whisper results
    if openai_api_key:
        print("\nRunning Whisper text search...")
        whisper_results = whisper_processor.test_search(text_query)
        
        print("\nSaving Whisper search results...")
        for i, result in enumerate(whisper_results):
            file_path = os.path.join(whisper_dir, f"result_{i+1}.txt")
            save_path = save_text_chunk(result, file_path)
            print(f"  Saved result {i+1} to {save_path}")
    else:
        print("\nSkipping Whisper search (no API key found)")
    
    # Run and save Wav2Vec results
    print("\nRunning Wav2Vec audio search...")
    wav2vec_results = wav2vec_processor.test_search(audio_segment)
    
    print("\nSaving Wav2Vec search results...")
    for i, result in enumerate(wav2vec_results):
        metadata_path = save_audio_chunk(result, audio_file, wav2vec_dir, i, "wav2vec")
        print(f"  Saved Wav2Vec result {i+1} metadata to {metadata_path}")
    
    # Run and save CLAP text-to-audio results
    print("\nRunning CLAP text-to-audio search...")
    clap_text_results = clap_processor.test_search_by_text(text_query)
    
    print("\nSaving CLAP text search results...")
    for i, result in enumerate(clap_text_results):
        metadata_path = save_audio_chunk(result, audio_file, clap_text_dir, i, "clap_text")
        print(f"  Saved CLAP text result {i+1} metadata to {metadata_path}")
    
    # Run and save CLAP audio-to-audio results
    print("\nRunning CLAP audio-to-audio search...")
    clap_audio_results = clap_processor.test_search_by_audio(audio_segment)
    
    print("\nSaving CLAP audio search results...")
    for i, result in enumerate(clap_audio_results):
        metadata_path = save_audio_chunk(result, audio_file, clap_audio_dir, i, "clap_audio")
        print(f"  Saved CLAP audio result {i+1} metadata to {metadata_path}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE - RESULTS SAVED")
    print("="*80)
    print(f"\nResults saved to the '{results_dir}' directory:")
    print(f"  - Whisper text search: {whisper_dir}")
    print(f"  - Wav2Vec audio search: {wav2vec_dir}")
    print(f"  - CLAP text-to-audio search: {clap_text_dir}")
    print(f"  - CLAP audio-to-audio search: {clap_audio_dir}")

if __name__ == "__main__":
    main() 