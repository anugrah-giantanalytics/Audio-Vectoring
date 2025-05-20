#!/bin/bash
# Run tests for the audio_vectoring package

echo "Running import tests..."
poetry run python tests/test_imports.py

if [ $? -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Tests failed!"
    exit 1
fi 