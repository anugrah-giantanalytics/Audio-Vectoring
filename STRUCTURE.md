# Audio Vectoring Project Structure

This document outlines the structure of the modular audio vectoring codebase.

## Core Package Structure

```
audio_vectoring/
├── processors/              # Audio processor implementations
│   ├── __init__.py          # Module initialization
│   ├── base.py              # Base processor interface
│   ├── whisper_processor.py # Whisper (text-based) processor
│   ├── wav2vec_processor.py # Wav2Vec (audio-based) processor
│   └── clap_processor.py    # CLAP (multimodal) processor
├── chunking/                # Audio and text chunking strategies
│   ├── __init__.py          # Module initialization
│   ├── text_chunking.py     # Text chunking methods
│   └── audio_chunking.py    # Advanced audio chunking methods
├── embeddings/              # Embedding functions
│   ├── __init__.py          # Module initialization
│   └── clap_embedding.py    # CLAP embedding functions
├── storage/                 # Vector database connectors
│   ├── __init__.py          # Module initialization
│   └── qdrant_connector.py  # Qdrant vector DB connector
└── utils/                   # Utility functions
    ├── __init__.py          # Module initialization
    └── audio_utils.py       # Audio processing utilities
```

## Scripts and Examples

```
scripts/
├── compare_search.py        # Script to run comparison search with all methods
└── run_tests.sh             # Script to run all tests

examples/
└── basic_usage.py           # Examples showing how to use each processor
```

## Tests

```
tests/
├── __init__.py              # Tests package initialization
└── test_imports.py          # Test that imports work correctly
```

## File Descriptions

### Core Modules

#### Processors

- `processors/base.py`: Abstract base class for all audio processors, defining the common interface.
- `processors/whisper_processor.py`: Implementation of the Whisper-based (text-based) audio processor using OpenAI's Whisper model.
- `processors/wav2vec_processor.py`: Implementation of the Wav2Vec-based (direct audio) processor for audio embedding.
- `processors/clap_processor.py`: Implementation of the CLAP-based (multimodal) processor supporting both text-to-audio and audio-to-audio search.

#### Chunking

- `chunking/text_chunking.py`: Functions for chunking text, including fixed-size, recursive, and sentence-based methods.
- `chunking/audio_chunking.py`: Functions for advanced audio chunking, including spectrogram and semantic shift detection.

#### Embeddings

- `embeddings/clap_embedding.py`: Implementation of CLAP embedding functions for both audio and text.

#### Storage

- `storage/qdrant_connector.py`: Connector for the Qdrant vector database with support for different vector dimensions.

#### Utils

- `utils/audio_utils.py`: Utility functions for audio processing, including loading, saving, and basic chunking.

### Scripts

- `scripts/compare_search.py`: Script to run a comparison search across all three methods (Whisper, Wav2Vec, CLAP).
- `scripts/run_tests.sh`: Bash script to run all tests.

### Examples

- `examples/basic_usage.py`: Examples showing how to use each processor with explanations.

### Tests

- `tests/test_imports.py`: Test script to ensure all imports work correctly.

## Data Directories

```
chunks/                      # Storage for processed chunks
└── whisper/                 # Whisper transcription chunks
    └── fallback/            # Fallback storage if vector DB fails

chunk_results/               # Storage for search results
├── whisper/                 # Whisper search results
├── wav2vec/                 # Wav2Vec search results
├── clap_text/               # CLAP text-to-audio search results
├── clap_audio/              # CLAP audio-to-audio search results
└── comparison_summary.md    # Comparison of results across methods
```

## Configuration Files

- `pyproject.toml`: Poetry project configuration with dependencies.
- `README.md`: Project overview and usage instructions.

## Legacy Files

The original files have been moved to the `legacy/` directory for reference:

```
legacy/
├── whisper.py               # Original Whisper implementation
├── wave2vec.py              # Original Wav2Vec implementation
├── clap.py                  # Original CLAP implementation
├── compare_search.py        # Original comparison script
├── compare_and_save.py      # Original comparison and save script
├── qdrant_setup.py          # Original Qdrant setup code
├── update_qdrant.py         # Original Qdrant update code
├── test_qdrant.py           # Original Qdrant test code
├── save_chunks.py           # Original chunk saving code
└── true_multimodal_rag.py   # Original multimodal RAG implementation
```

These files contain the original functionality that has been refactored into the modular architecture.
