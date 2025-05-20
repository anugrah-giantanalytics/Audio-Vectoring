#!/bin/bash

# Clean up cache directories
echo "Cleaning up cache directories..."
rm -rf chunks/* chunk_results/* __pycache__/*

# Create directories if they don't exist
mkdir -p chunks/whisper chunks/clap chunks/wav2vec
mkdir -p chunk_results/whisper chunk_results/clap_text chunk_results/clap_audio chunk_results/wav2vec

# Set environment variables
# Actual Qdrant Cloud and OpenAI credentials
echo "Setting up environment variables..."
export OPENAI_API_KEY="your-openai-api-key"
export QDRANT_URL="your-qdrant-url " 
export QDRANT_API_KEY="your-qdrant-api-key"

echo "Environment setup complete!"
echo ""
echo "Credentials set and ready to process large audio files." 