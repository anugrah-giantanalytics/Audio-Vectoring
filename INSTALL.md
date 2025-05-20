# Installation Guide

This guide walks you through setting up the Audio Vectorization Framework on your system.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Python 3.9 or later** (but less than 3.11)

   ```bash
   python --version
   ```

2. **Poetry** (for dependency management)

   ```bash
   # Install Poetry (if not already installed)
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **FFmpeg** (required for audio processing)

   ```bash
   # For macOS (using Homebrew)
   brew install ffmpeg

   # For Ubuntu/Debian
   sudo apt-get update && sudo apt-get install ffmpeg

   # For Windows (using Chocolatey)
   choco install ffmpeg
   ```

## Installation Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd audioVectoring
   ```

2. **Install dependencies with Poetry**

   ```bash
   poetry install
   ```

3. **Verify the installation**

   ```bash
   poetry run python tests/test_imports.py
   ```

   You should see: `All imports successful!`

## Optional: OpenAI API Key Setup

For the Whisper-based processor, you'll need an OpenAI API key:

1. **Get an API key** from [OpenAI](https://platform.openai.com/api-keys)

2. **Set the environment variable**

   ```bash
   # For macOS/Linux
   export OPENAI_API_KEY=your-api-key-here

   # For Windows
   set OPENAI_API_KEY=your-api-key-here
   ```

## Running Examples

Once installation is complete, you can run the example script:

```bash
# If you set up an OpenAI API key
poetry run python examples/basic_usage.py

# Without OpenAI API key (will skip Whisper examples)
poetry run python examples/basic_usage.py
```

## Running the Comparison Search

To run a comparison search across all three methods:

```bash
# If you set up an OpenAI API key
poetry run python scripts/compare_search.py

# Without OpenAI API key (will skip Whisper search)
poetry run python scripts/compare_search.py
```

## Developing

If you're developing the codebase:

1. **Activate the Poetry virtual environment**

   ```bash
   poetry shell
   ```

2. **Run tests**

   ```bash
   ./scripts/run_tests.sh
   ```

3. **Format code**
   ```bash
   poetry run black audio_vectoring
   poetry run isort audio_vectoring
   ```

## Troubleshooting

- **ImportError: No module named 'audio_vectoring'**: Ensure you're running Python through Poetry (`poetry run python`) or have activated the Poetry shell (`poetry shell`).

- **RuntimeWarning about ffmpeg**: Make sure ffmpeg is installed and in your PATH.

- **OpenAI API errors**: Check that your API key is correctly set and has sufficient permissions.

## Additional Resources

- Read `README.md` for usage information
- See `STRUCTURE.md` for codebase structure details
- Check `examples/basic_usage.py` for detailed usage examples
