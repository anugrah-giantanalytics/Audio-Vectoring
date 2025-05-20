# Audio Vectorization Framework

This project implements and compares different approaches for audio vectorization and search:

1. **Whisper-based** (Text-based): Transcribes audio to text, then vectorizes the text
2. **Wav2Vec-based** (Direct Audio): Directly embeds audio segments
3. **CLAP-based** (Multimodal): Uses a multimodal model for both text-to-audio and audio-to-audio search

## ⚠️ Code Restructuring Notice

> This codebase has been restructured from its original form to a more modular architecture.
> The original files (whisper.py, wave2vec.py, clap.py, etc.) remain for reference, but all
> functionality has been refactored into the `audio_vectoring` package.
> For more details on the structure, see `STRUCTURE.md`.

## Project Structure

```
audio_vectoring/
├── processors/              # Audio processor implementations
│   ├── base.py              # Base processor interface
│   ├── whisper_processor.py # Whisper (text-based) processor
│   ├── wav2vec_processor.py # Wav2Vec (audio-based) processor
│   └── clap_processor.py    # CLAP (multimodal) processor
├── chunking/                # Audio and text chunking strategies
│   ├── text_chunking.py     # Text chunking methods
│   └── audio_chunking.py    # Advanced audio chunking methods
├── embeddings/              # Embedding functions
│   └── clap_embedding.py    # CLAP embedding functions
├── storage/                 # Vector database connectors
│   └── qdrant_connector.py  # Qdrant vector DB connector
└── utils/                   # Utility functions
    └── audio_utils.py       # Audio processing utilities

scripts/
└── compare_search.py        # Script to run comparison search

chunks/                      # Storage for chunks
chunk_results/               # Storage for search results
```

## Requirements

- Python 3.8+
- Poetry for dependency management
- OpenAI API key (for Whisper-based approach)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository
2. Install dependencies with Poetry:
   ```
   poetry install
   ```

## Usage

### Running a comparison search

```bash
# Set OpenAI API key if using Whisper
export OPENAI_API_KEY=your_api_key_here

# Run comparison search
poetry run python scripts/compare_search.py
```

This will:

1. Process the test audio file with all three methods
2. Run searches using the same query across all methods
3. Save results to the `chunk_results` directory
4. Generate a comparison summary

### Customizing the search

You can modify `scripts/compare_search.py` to:

- Change the audio file
- Modify the text query
- Adjust chunking methods and parameters
- Customize the comparison metrics

## Chunking Strategies

Each processor supports different chunking strategies:

### Whisper (Text-based)

- `fixed_text`: Fixed-size text chunks
- `recursive`: Recursive character splitting with optimized separators
- `sentences`: Sentence-based chunking

### Wav2Vec (Audio-based)

- `fixed`: Fixed-size audio chunks
- `silence`: Silence detection
- `spectrogram`: Spectrogram-based chunking

### CLAP (Multimodal)

- `fixed`: Fixed-size audio chunks
- `silence`: Silence detection
- `semantic`: Semantic shift detection

## Search Capabilities

- **Whisper**: Text-based semantic search
- **Wav2Vec**: Audio similarity search
- **CLAP**: Both text-to-audio and audio-to-audio search

## Benchmark Results

### Whisper Approach (Text-based)

```
Transcription Quality WER: 0.0556
Transcription Quality CER: 0.0115

Chunking Results:
Method                    Chunks     Avg Length      Time (s)
------------------------------------------------------------
Fixed (100 chars)         55         96.9            0.001
Fixed (500 chars)         1          447.0           0.000
Fixed (1000 chars)        1          447.0           0.000
Recursive (500 chars)     1          447.0           0.000
```

### Wav2Vec Approach (Direct Audio)

```
Wav2Vec Chunking Results:
Method                    Chunks     Chunk Time (s)  Embed Time (s)  Total Time (s)
--------------------------------------------------------------------------------
Fixed (2 sec)             34         0.000           1.074           1.074
Fixed (5 sec)             14         0.000           0.210           0.210
Fixed (10 sec)            7          0.000           0.372           0.372
Spectrogram (1000 ms)     38         1.440           0.060           1.501

Embedding shape: (768,)
```

### CLAP Approach (Multimodal)

```
CLAP Chunking Results:
Method                    Chunks     Chunk Time (s)  Audio Embed (s) Text Embed (s)  Total (s)
------------------------------------------------------------------------------------------
Fixed (5 sec)             14         0.000           0.279           0.944           1.223
Fixed (10 sec)            7          0.000           0.521           0.059           0.580
Semantic (threshold=25%)  4          3.664           0.149           0.048           3.861

Embedding shape: (512,)
```

## Comparison of Approaches

| Approach    | Pros                                                                                        | Cons                                                         | Best Use Cases                                              |
| ----------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| **Whisper** | - Good semantic understanding<br>- Text-based search<br>- Smaller storage requirements      | - Depends on transcription quality<br>- Loses audio nuances  | - Speech-heavy content<br>- When text search is important   |
| **Wav2Vec** | - Direct audio similarity<br>- No transcription needed<br>- Preserves audio characteristics | - No semantic understanding<br>- Larger storage requirements | - Music similarity<br>- Sound effects<br>- Non-speech audio |
| **CLAP**    | - Multimodal (text + audio)<br>- Semantic understanding<br>- More robust to noise           | - More complex<br>- Slower processing                        | - Mixed content<br>- When flexibility is needed             |

## Using Persistent Storage with Qdrant

By default, the implementations use in-memory Qdrant collections that are lost when the process ends. To use persistent storage:

### 1. Local Qdrant Instance (Docker)

```bash
# Install and run Qdrant locally with Docker
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Update your code to use the local instance:

```python
from qdrant_setup import QdrantConnector

# Connect to local Qdrant instance
qdrant_connector = QdrantConnector(url="http://localhost:6333")
```

### 2. Qdrant Cloud

Sign up at [https://cloud.qdrant.io/](https://cloud.qdrant.io/) to get a URL and API key.

```python
from qdrant_setup import QdrantConnector

# Connect to Qdrant Cloud
qdrant_connector = QdrantConnector(
    url="https://your-qdrant-cloud-instance.qdrant.tech",
    api_key="your-api-key-here"
)
```
