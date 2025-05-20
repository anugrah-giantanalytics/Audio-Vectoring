# Audio Vectorization Commands

## Environment Setup

```bash
# Set up environment variables and clean cache
./setup_env.sh

# Set Qdrant Cloud and OpenAI credentials directly
export QDRANT_URL="your-qdrant-url"
export QDRANT_API_KEY="your-qdrant-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Running Comparison Scripts

```bash
# Basic comparison with default parameters
poetry run python compare_search.py

# Process with specific chunking method and parameter
poetry run python compare_search.py --chunking_method recursive --chunk_param 500

# Process only with Whisper (skip other methods)
poetry run python compare_search.py --skip_wav2vec --skip_clap

# Process a different audio file
poetry run python compare_search.py --audio_file path/to/your/audio.wav

# Custom text query
poetry run python compare_search.py --text_query "your search query here"
```

## Testing Qdrant Connection

```bash
# Test Qdrant connection
poetry run python test_qdrant.py

# Update Qdrant collections
poetry run python update_qdrant.py
```

## Cleaning Up Cache

```bash
# Manual cleanup
rm -rf chunks/* chunk_results/* __pycache__/*
```

# Example command (use placeholders for actual keys)

export OPENAI_API_KEY="your-openai-api-key" && export QDRANT_URL="your-qdrant-url" && export QDRANT_API_KEY="your-qdrant-api-key" && poetry run python -c "from whisper import WhisperAudioProcessor; p = WhisperAudioProcessor(api_key='$OPENAI_API_KEY'); p.test_search('foreign trade zone manual');"
