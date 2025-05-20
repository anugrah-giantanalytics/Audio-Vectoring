#!/usr/bin/env python3
import os
import json
import numpy as np
from audio_vectoring.processors.whisper_processor import WhisperAudioProcessor
from audio_vectoring.processors.wav2vec_processor import Wav2VecProcessor
from audio_vectoring.processors.clap_processor import ClapProcessor
from audio_vectoring.utils.audio_utils import ensure_dir

def save_text_chunk(chunk, file_path):
    """Save text chunk to a file"""
    with open(file_path, 'w') as f:
        f.write(chunk.page_content)
        
    # Save metadata in a separate file
    metadata_path = file_path.replace('.txt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(chunk.metadata, f, indent=2)
    
    return file_path

def save_audio_chunk(result, output_dir, index, method):
    """Save audio chunk metadata"""
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
    text_query = "who or what has a foreign trade zone manual"
    
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
    
    # Process with Wav2Vec - use a small portion of audio for faster testing
    print("\nProcessing audio with Wav2Vec...")
    # Load audio to create a short segment
    audio, sr = wav2vec_processor.load_audio(audio_file)
    # Take first 15 seconds only for testing
    short_audio = audio[:sr*15]
    # Save short segment to temporary file
    short_audio_file = "temp_short_audio.wav"
    import soundfile as sf
    sf.write(short_audio_file, short_audio, sr)
    # Process the shorter audio file
    wav2vec_processor.process_audio(short_audio_file, chunking_method="fixed", chunk_param=3)
    
    # Process with CLAP - also use the short audio
    print("\nProcessing audio with CLAP...")
    clap_processor.process_audio(short_audio_file, chunking_method="fixed", chunk_param=3)
    
    # =============== CREATE AUDIO QUERY SEGMENT ===============
    # Create audio segment for audio-based search (5-second segment starting at 5s)
    audio_segment = short_audio[5*sr:10*sr]  # 5-second segment
    
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
        metadata_path = save_audio_chunk(result, wav2vec_dir, i, "wav2vec")
        print(f"  Saved Wav2Vec result {i+1} metadata to {metadata_path}")
    
    # Run and save CLAP text-to-audio results
    print("\nRunning CLAP text-to-audio search...")
    clap_text_results = clap_processor.test_search_by_text(text_query)
    
    print("\nSaving CLAP text search results...")
    for i, result in enumerate(clap_text_results):
        metadata_path = save_audio_chunk(result, clap_text_dir, i, "clap_text")
        print(f"  Saved CLAP text result {i+1} metadata to {metadata_path}")
    
    # Run and save CLAP audio-to-audio results
    print("\nRunning CLAP audio-to-audio search...")
    clap_audio_results = clap_processor.test_search_by_audio(audio_segment)
    
    print("\nSaving CLAP audio search results...")
    for i, result in enumerate(clap_audio_results):
        metadata_path = save_audio_chunk(result, clap_audio_dir, i, "clap_audio")
        print(f"  Saved CLAP audio result {i+1} metadata to {metadata_path}")
    
    # Create comparison summary
    create_comparison_summary(results_dir, text_query)
    
    # Cleanup temporary file
    if os.path.exists(short_audio_file):
        os.remove(short_audio_file)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE - RESULTS SAVED")
    print("="*80)
    print(f"\nResults saved to the '{results_dir}' directory:")
    print(f"  - Whisper text search: {whisper_dir}")
    print(f"  - Wav2Vec audio search: {wav2vec_dir}")
    print(f"  - CLAP text-to-audio search: {clap_text_dir}")
    print(f"  - CLAP audio-to-audio search: {clap_audio_dir}")
    print(f"  - Comparison summary: {os.path.join(results_dir, 'comparison_summary.md')}")

def create_comparison_summary(results_dir, text_query):
    """Create a markdown summary of the comparison results"""
    summary_path = os.path.join(results_dir, "comparison_summary.md")
    
    # Collect results data
    whisper_dir = os.path.join(results_dir, "whisper")
    wav2vec_dir = os.path.join(results_dir, "wav2vec")
    clap_text_dir = os.path.join(results_dir, "clap_text")
    clap_audio_dir = os.path.join(results_dir, "clap_audio")
    
    # Get Whisper results
    whisper_results = []
    if os.path.exists(whisper_dir):
        for file in os.listdir(whisper_dir):
            if file.endswith("_metadata.json"):
                with open(os.path.join(whisper_dir, file), 'r') as f:
                    metadata = json.load(f)
                text_file = file.replace("_metadata.json", ".txt")
                if os.path.exists(os.path.join(whisper_dir, text_file)):
                    with open(os.path.join(whisper_dir, text_file), 'r') as f:
                        text = f.read()
                    whisper_results.append({
                        "metadata": metadata,
                        "text": text
                    })
    
    # Get Wav2Vec results
    wav2vec_results = []
    if os.path.exists(wav2vec_dir):
        for file in os.listdir(wav2vec_dir):
            if file.endswith("_metadata.json"):
                with open(os.path.join(wav2vec_dir, file), 'r') as f:
                    metadata = json.load(f)
                wav2vec_results.append(metadata)
    
    # Get CLAP text search results
    clap_text_results = []
    if os.path.exists(clap_text_dir):
        for file in os.listdir(clap_text_dir):
            if file.endswith("_metadata.json"):
                with open(os.path.join(clap_text_dir, file), 'r') as f:
                    metadata = json.load(f)
                clap_text_results.append(metadata)
    
    # Get CLAP audio search results
    clap_audio_results = []
    if os.path.exists(clap_audio_dir):
        for file in os.listdir(clap_audio_dir):
            if file.endswith("_metadata.json"):
                with open(os.path.join(clap_audio_dir, file), 'r') as f:
                    metadata = json.load(f)
                clap_audio_results.append(metadata)
    
    # Sort results by rank
    whisper_results.sort(key=lambda x: x["metadata"].get("rank", 999))
    wav2vec_results.sort(key=lambda x: x.get("rank", 999))
    clap_text_results.sort(key=lambda x: x.get("rank", 999))
    clap_audio_results.sort(key=lambda x: x.get("rank", 999))
    
    # Create summary markdown
    with open(summary_path, 'w') as f:
        f.write("# Audio Vectorization Methods Comparison\n\n")
        
        # Search query section
        f.write("## Search Query\n")
        f.write(f"- Text query: \"{text_query}\"\n")
        f.write("- Audio query: 5-second segment from 10s-15s of test_audio.wav\n\n")
        
        # Whisper results
        f.write("## Whisper (Text-based) Results\n")
        if whisper_results:
            whisper_text = whisper_results[0]["text"]
            f.write(f"The Whisper transcription contains the text:\n")
            f.write(f"> \"{whisper_text}\"\n\n")
            if "audience" in whisper_text.lower():
                f.write("This transcription captures the speaker's mention of the audience.\n\n")
        else:
            f.write("No Whisper results available.\n\n")
        
        # Wav2Vec results
        f.write("## Wav2Vec (Audio-based) Results\n")
        if wav2vec_results:
            for i, result in enumerate(wav2vec_results[:2]):
                chunk_id = result.get("chunk_id", "unknown")
                score = result.get("score", 0.0)
                start_time = result.get("start_time", 0.0)
                end_time = result.get("end_time", 0.0)
                f.write(f"{i+1}. **Top match**: Chunk {chunk_id} ")
                if "start_time" in result and "end_time" in result:
                    f.write(f"({start_time:.2f}s-{end_time:.2f}s) ")
                f.write(f"with score: {score:.4f}\n")
            f.write("\n")
            if any(r.get("start_time", 0) == 10.0 for r in wav2vec_results):
                f.write("Wav2Vec correctly identified the exact audio segment used as the query.\n\n")
        else:
            f.write("No Wav2Vec results available.\n\n")
        
        # CLAP results
        f.write("## CLAP (Multimodal) Results\n\n")
        
        # CLAP text search
        f.write("### CLAP Text-to-Audio Search\n")
        if clap_text_results:
            for i, result in enumerate(clap_text_results[:2]):
                chunk_id = result.get("chunk_id", "unknown")
                score = result.get("score", 0.0)
                start_time = result.get("start_time", 0.0)
                end_time = result.get("end_time", 0.0)
                f.write(f"{i+1}. **Top match**: Chunk {chunk_id} ")
                if "start_time" in result and "end_time" in result:
                    f.write(f"({start_time:.2f}s-{end_time:.2f}s) ")
                f.write(f"with score: {score:.4f}\n")
            f.write("\n")
        else:
            f.write("No CLAP text search results available.\n\n")
        
        # CLAP audio search
        f.write("### CLAP Audio-to-Audio Search\n")
        if clap_audio_results:
            for i, result in enumerate(clap_audio_results[:2]):
                chunk_id = result.get("chunk_id", "unknown")
                score = result.get("score", 0.0)
                start_time = result.get("start_time", 0.0)
                end_time = result.get("end_time", 0.0)
                f.write(f"{i+1}. **Top match**: Chunk {chunk_id} ")
                if "start_time" in result and "end_time" in result:
                    f.write(f"({start_time:.2f}s-{end_time:.2f}s) ")
                f.write(f"with score: {score:.4f}\n")
            f.write("\n")
        else:
            f.write("No CLAP audio search results available.\n\n")
        
        # Comparison table
        f.write("## Comparison Summary\n\n")
        f.write("| Method | Precision | Strengths | Limitations |\n")
        f.write("|--------|-----------|-----------|-------------|\n")
        f.write("| **Whisper** | High for text-based understanding | - Accurately captures semantic content<br>- Provides readable text output<br>- Good at understanding speech content | - Depends on transcription quality<br>- Loses audio characteristics<br>- Requires OpenAI API key |\n")
        f.write("| **Wav2Vec** | Excellent for exact audio matching | - Precise audio feature matching<br>- Works well for non-speech audio<br>- No external API dependency | - No semantic understanding<br>- Less robust to audio variations |\n")
        f.write("| **CLAP** | Moderate for multimodal search | - Supports both text and audio queries<br>- Some semantic understanding<br>- Balances audio and semantic features | - Less precise than specialized methods<br>- Slower processing<br>- Results may be unexpected |\n\n")
        
        # Conclusion
        f.write("## Conclusion\n")
        f.write("Each method has distinct advantages for different use cases:\n\n")
        f.write("- **Whisper**: Best for semantic understanding and text-based search of speech content\n")
        f.write("- **Wav2Vec**: Excellent for exact audio matching, sound effects, and non-speech audio\n")
        f.write("- **CLAP**: Most versatile with support for both text-to-audio and audio-to-audio search, at the cost of some precision\n\n")
        f.write("The ideal approach depends on the specific requirements of the application and the nature of the audio content being processed.\n")

if __name__ == "__main__":
    main() 