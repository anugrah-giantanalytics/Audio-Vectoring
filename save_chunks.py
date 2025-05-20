import os
import json
import numpy as np
import soundfile as sf
from typing import Dict, List, Any
import pandas as pd

from whisper import WhisperAudioProcessor
from wave2vec import Wav2VecProcessor
from clap import ClapProcessor

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_text_chunk(chunk_content, metadata, output_dir, index):
    """Save text chunk to file"""
    ensure_dir(output_dir)
    
    # Save content
    with open(f"{output_dir}/chunk_{index}.txt", "w") as f:
        f.write(chunk_content)
    
    # Save metadata
    with open(f"{output_dir}/chunk_{index}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def save_audio_chunk(audio_data, metadata, output_dir, index, sample_rate=16000):
    """Save audio chunk to file"""
    ensure_dir(output_dir)
    
    # Save audio
    sf.write(f"{output_dir}/chunk_{index}.wav", audio_data, sample_rate)
    
    # Save metadata
    with open(f"{output_dir}/chunk_{index}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def save_whisper_chunks(processor, audio_file, output_dir="chunks/whisper"):
    """Process audio with Whisper and save the chunks"""
    print(f"Processing {audio_file} with Whisper and saving chunks to {output_dir}...")
    
    # Process with different chunking methods
    chunking_methods = [
        {"name": "fixed_text", "param": 200, "dir": f"{output_dir}/fixed_200"},
        {"name": "fixed_text", "param": 500, "dir": f"{output_dir}/fixed_500"},
        {"name": "recursive", "param": 500, "dir": f"{output_dir}/recursive_500"}
    ]
    
    # Load and transcribe audio
    audio, sr = processor.load_audio(audio_file)
    transcription = processor.transcribe_audio(audio)
    
    summary = []
    
    # Process with each method
    for method in chunking_methods:
        method_dir = method["dir"]
        ensure_dir(method_dir)
        
        # Save full transcription
        with open(f"{method_dir}/full_transcription.txt", "w") as f:
            f.write(transcription)
        
        print(f"  Chunking with {method['name']} (param={method['param']})...")
        
        # Generate chunks based on method
        if method["name"] == "fixed_text":
            chunks = processor.chunk_text_fixed_size(transcription, chunk_size=method["param"])
        elif method["name"] == "recursive":
            chunks = processor.chunk_text_recursive(transcription, chunk_size=method["param"])
        else:
            continue
        
        # Save each chunk
        for i, chunk in enumerate(chunks):
            save_text_chunk(
                chunk.page_content,
                {"index": i, "method": method["name"], "param": method["param"], **chunk.metadata},
                method_dir,
                i
            )
        
        # Add to summary
        summary.append({
            "method": f"{method['name']}_{method['param']}",
            "chunks": len(chunks),
            "avg_length": sum(len(doc.page_content) for doc in chunks) / len(chunks) if chunks else 0
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    print(f"  Saved Whisper chunks summary to {output_dir}/summary.csv")
    print(f"  Whisper processing complete, created {sum(item['chunks'] for item in summary)} total chunks")
    
    return summary

def save_wav2vec_chunks(processor, audio_file, output_dir="chunks/wav2vec"):
    """Process audio with Wav2Vec and save the chunks"""
    print(f"Processing {audio_file} with Wav2Vec and saving chunks to {output_dir}...")
    
    # Process with different chunking methods
    chunking_methods = [
        {"name": "fixed", "param": 2, "dir": f"{output_dir}/fixed_2s"},
        {"name": "fixed", "param": 5, "dir": f"{output_dir}/fixed_5s"},
        {"name": "spectrogram", "param": 1000, "dir": f"{output_dir}/spectrogram_1000ms"}
    ]
    
    # Load audio
    audio, sr = processor.load_audio(audio_file)
    
    summary = []
    
    # Process with each method
    for method in chunking_methods:
        method_dir = method["dir"]
        ensure_dir(method_dir)
        
        print(f"  Chunking with {method['name']} (param={method['param']})...")
        
        # Generate chunks based on method
        if method["name"] == "fixed":
            chunks = processor.chunk_audio_fixed_size(audio, sr, chunk_duration_sec=method["param"])
            # Generate metadata
            metadatas = [
                {"chunk_id": i, "start_time": i * method["param"], "end_time": (i + 1) * method["param"]} 
                for i in range(len(chunks))
            ]
        elif method["name"] == "spectrogram":
            chunks = processor.chunk_audio_by_spectrogram(audio, sr, segment_ms=method["param"])
            # For spectrogram chunking, we need to generate metadata differently
            metadatas = []
            current_pos = 0
            for i, chunk in enumerate(chunks):
                start_time = current_pos / sr
                end_time = (current_pos + len(chunk)) / sr
                metadatas.append({
                    "chunk_id": i,
                    "start_time": start_time,
                    "end_time": end_time
                })
                current_pos += len(chunk)
        else:
            continue
        
        # Save each chunk
        for i, chunk in enumerate(chunks):
            metadata = metadatas[i] if i < len(metadatas) else {"chunk_id": i}
            save_audio_chunk(
                chunk,
                {"method": method["name"], "param": method["param"], **metadata},
                method_dir,
                i,
                sr
            )
        
        # Add to summary
        summary.append({
            "method": f"{method['name']}_{method['param']}",
            "chunks": len(chunks),
            "avg_duration": sum(len(chunk)/sr for chunk in chunks) / len(chunks) if chunks else 0
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    print(f"  Saved Wav2Vec chunks summary to {output_dir}/summary.csv")
    print(f"  Wav2Vec processing complete, created {sum(item['chunks'] for item in summary)} total chunks")
    
    return summary

def save_clap_chunks(processor, audio_file, output_dir="chunks/clap"):
    """Process audio with CLAP and save the chunks"""
    print(f"Processing {audio_file} with CLAP and saving chunks to {output_dir}...")
    
    # Process with different chunking methods
    chunking_methods = [
        {"name": "fixed", "param": 5, "dir": f"{output_dir}/fixed_5s"},
        {"name": "fixed", "param": 10, "dir": f"{output_dir}/fixed_10s"},
        {"name": "semantic", "param": 25, "dir": f"{output_dir}/semantic_25pct"}
    ]
    
    # Load audio
    audio, sr = processor.load_audio(audio_file)
    
    summary = []
    
    # Process with each method
    for method in chunking_methods:
        method_dir = method["dir"]
        ensure_dir(method_dir)
        
        print(f"  Chunking with {method['name']} (param={method['param']})...")
        
        try:
            # Generate chunks based on method
            if method["name"] == "fixed":
                chunks = processor.chunk_audio_fixed_size(audio, sr, chunk_duration_sec=method["param"])
                # Generate metadata
                metadatas = [
                    {"chunk_id": i, "start_time": i * method["param"], "end_time": (i + 1) * method["param"]} 
                    for i in range(len(chunks))
                ]
                
                # Save each chunk
                for i, chunk in enumerate(chunks):
                    metadata = metadatas[i] if i < len(metadatas) else {"chunk_id": i}
                    save_audio_chunk(
                        chunk,
                        {"method": method["name"], "param": method["param"], **metadata},
                        method_dir,
                        i,
                        sr
                    )
                
                # Add to summary
                summary.append({
                    "method": f"{method['name']}_{method['param']}",
                    "chunks": len(chunks),
                    "avg_duration": sum(len(chunk)/sr for chunk in chunks) / len(chunks) if chunks else 0
                })
                
            elif method["name"] == "semantic":
                semantic_chunks = processor.chunk_audio_by_semantic_shift(
                    audio, sr, threshold=method["param"]/100.0
                )
                
                # For semantic chunking, chunks are dictionaries with 'audio' key
                for i, chunk in enumerate(semantic_chunks):
                    metadata = {
                        "chunk_id": i,
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "method": method["name"],
                        "param": method["param"]
                    }
                    save_audio_chunk(
                        chunk["audio"],
                        metadata,
                        method_dir,
                        i,
                        sr
                    )
                
                # Add to summary
                summary.append({
                    "method": f"{method['name']}_{method['param']}",
                    "chunks": len(semantic_chunks),
                    "avg_duration": sum((chunk["end_time"] - chunk["start_time"]) 
                                     for chunk in semantic_chunks) / len(semantic_chunks) if semantic_chunks else 0
                })
                
        except Exception as e:
            print(f"  Error processing {method['name']} method: {e}")
    
    # Save summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
        print(f"  Saved CLAP chunks summary to {output_dir}/summary.csv")
        print(f"  CLAP processing complete, created {sum(item['chunks'] for item in summary)} total chunks")
    else:
        print("  No CLAP chunks were created successfully")
    
    return summary

def main():
    """Main function to save chunks from all methods"""
    # Set OpenAI API key (should be already set in environment)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        print("WARNING: No OpenAI API key found. Whisper processing will be limited.")
    
    # Audio file to process
    audio_file = "test_audio.wav"
    
    # Process with each method
    results = {}
    
    # Initialize processors
    print("Initializing processors...")
    whisper_processor = WhisperAudioProcessor(api_key=openai_api_key)
    wav2vec_processor = Wav2VecProcessor()
    clap_processor = ClapProcessor()
    
    # Save chunks for each method
    results["whisper"] = save_whisper_chunks(whisper_processor, audio_file)
    results["wav2vec"] = save_wav2vec_chunks(wav2vec_processor, audio_file)
    results["clap"] = save_clap_chunks(clap_processor, audio_file)
    
    # Save overall summary
    print("\nSaving overall summary...")
    overall_summary = {
        "whisper_chunks": sum(item["chunks"] for item in results["whisper"]),
        "wav2vec_chunks": sum(item["chunks"] for item in results["wav2vec"]),
        "clap_chunks": sum(item["chunks"] for item in results["clap"]),
    }
    
    with open("chunks/summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)
    
    print("\nChunk saving complete!")
    print(f"Total chunks created:")
    print(f"  Whisper: {overall_summary['whisper_chunks']}")
    print(f"  Wav2Vec: {overall_summary['wav2vec_chunks']}")
    print(f"  CLAP: {overall_summary['clap_chunks']}")
    print("\nChunks saved to 'chunks/' directory")

if __name__ == "__main__":
    main() 