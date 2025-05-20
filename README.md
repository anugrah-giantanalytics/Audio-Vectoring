# Audio Vectorization Approaches

This project implements and compares three different approaches for audio vectorization and search using Qdrant as the vector database:

1. **Whisper-based** (Text-based): Transcribes audio to text, then chunks and embeds the text
2. **Wav2Vec-based** (Direct Audio): Directly embeds audio segments without transcription
3. **CLAP-based** (Multimodal): Supports both text-to-audio and audio-to-audio search

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

## Getting Started

1. Install dependencies:

```bash
poetry install
```

2. Set up environment and clean caches:

```bash
# First edit the script to add your API keys
nano setup_env.sh

# Then run it
./setup_env.sh
```

3. Run any of the approaches:

```bash
poetry run python whisper.py
poetry run python wave2vec.py
poetry run python clap.py
```

## Processing Large Audio Files

When working with long audio files (1+ hour), follow these best practices:

1. **Use Qdrant Cloud**: Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/) and set your credentials in `setup_env.sh`
2. **Clean cache files**: Run `./setup_env.sh` to clear old chunks and cache files
3. **Use recursive chunking**: This method provides better context preservation:
   ```bash
   # Example for processing large files with Whisper
   poetry run python compare_search.py --chunking_method recursive --chunk_param 500
   ```
4. **Monitor resource usage**: These processes can be memory-intensive, especially for Whisper transcription

The updated code now handles long audio files by:

- Processing audio in manageable segments
- Using cloud storage for better performance
- Implementing more robust error handling

## Requirements

- Python 3.10+
- Poetry
- FFmpeg (for audio processing)
- OpenAI API key (optional, for Whisper embeddings)
- Qdrant Cloud account (recommended for large files)
