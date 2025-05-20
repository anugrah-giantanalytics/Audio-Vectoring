import os
from qdrant_setup import QdrantConnector
from typing import Optional


def update_whisper_implementation(qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None):
    """
    Helper function to update the Whisper implementation to use a specified Qdrant instance.
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: API key for Qdrant Cloud
    """
    with open('whisper.py', 'r') as file:
        content = file.read()
    
    # Replace in-memory Qdrant client with our Qdrant connector
    if qdrant_url:
        if qdrant_api_key:
            qdrant_init = f"""
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(url="{qdrant_url}", api_key="{qdrant_api_key}")
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "whisper_audio_collection"
        
        # Create collection
        qdrant_connector.create_whisper_collection(self.collection_name)
"""
        else:
            qdrant_init = f"""
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(url="{qdrant_url}")
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "whisper_audio_collection"
        
        # Create collection
        qdrant_connector.create_whisper_collection(self.collection_name)
"""
    else:
        qdrant_init = """
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(in_memory=True)
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "whisper_audio_collection"
        
        # Create collection
        qdrant_connector.create_whisper_collection(self.collection_name)
"""

    # Find and replace the original Qdrant initialization code
    content = content.replace("""        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(":memory:")  # In-memory for testing
        self.collection_name = "whisper_audio_collection"
        
        # Create collection
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embeddings dimension
                distance=models.Distance.COSINE,
            ),
        )""", qdrant_init.strip())
    
    with open('whisper.py', 'w') as file:
        file.write(content)
    
    print("Updated whisper.py to use the specified Qdrant instance")


def update_wave2vec_implementation(qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None):
    """
    Helper function to update the Wav2Vec implementation to use a specified Qdrant instance.
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: API key for Qdrant Cloud
    """
    with open('wave2vec.py', 'r') as file:
        content = file.read()
    
    # Replace in-memory Qdrant client with our Qdrant connector
    if qdrant_url:
        if qdrant_api_key:
            qdrant_init = f"""
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(url="{qdrant_url}", api_key="{qdrant_api_key}")
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "wav2vec_audio_collection"
        
        # Create collection
        qdrant_connector.create_wav2vec_collection(self.collection_name)
"""
        else:
            qdrant_init = f"""
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(url="{qdrant_url}")
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "wav2vec_audio_collection"
        
        # Create collection
        qdrant_connector.create_wav2vec_collection(self.collection_name)
"""
    else:
        qdrant_init = """
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(in_memory=True)
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "wav2vec_audio_collection"
        
        # Create collection
        qdrant_connector.create_wav2vec_collection(self.collection_name)
"""

    # Find and replace the original Qdrant initialization code
    content = content.replace("""        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(":memory:")  # In-memory for testing
        self.collection_name = "wav2vec_audio_collection"
        
        # Create collection - Wav2Vec hidden states are typically 768-dimensional
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=768,  # Wav2Vec embeddings dimension
                distance=models.Distance.COSINE,
            ),
        )""", qdrant_init.strip())
    
    with open('wave2vec.py', 'w') as file:
        file.write(content)
    
    print("Updated wave2vec.py to use the specified Qdrant instance")


def update_clap_implementation(qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None):
    """
    Helper function to update the CLAP implementation to use a specified Qdrant instance.
    
    Args:
        qdrant_url: URL of the Qdrant server
        qdrant_api_key: API key for Qdrant Cloud
    """
    with open('clap.py', 'r') as file:
        content = file.read()
    
    # Replace in-memory Qdrant client with our Qdrant connector
    if qdrant_url:
        if qdrant_api_key:
            qdrant_init = f"""
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(url="{qdrant_url}", api_key="{qdrant_api_key}")
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "clap_audio_collection"
        
        # Create collection
        qdrant_connector.create_clap_collection(self.collection_name)
"""
        else:
            qdrant_init = f"""
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(url="{qdrant_url}")
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "clap_audio_collection"
        
        # Create collection
        qdrant_connector.create_clap_collection(self.collection_name)
"""
    else:
        qdrant_init = """
        # Initialize Qdrant client
        from qdrant_setup import QdrantConnector
        qdrant_connector = QdrantConnector(in_memory=True)
        self.qdrant_client = qdrant_connector.client
        self.collection_name = "clap_audio_collection"
        
        # Create collection
        qdrant_connector.create_clap_collection(self.collection_name)
"""

    # Find and replace the original Qdrant initialization code
    content = content.replace("""        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(":memory:")  # In-memory for testing
        self.collection_name = "clap_audio_collection"
        
        # Create collection - CLAP embeddings are 512-dimensional
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=512,  # CLAP embeddings dimension
                distance=models.Distance.COSINE,
            ),
        )""", qdrant_init.strip())
    
    with open('clap.py', 'w') as file:
        file.write(content)
    
    print("Updated clap.py to use the specified Qdrant instance")


if __name__ == "__main__":
    # Example: Update all implementations to use a remote Qdrant instance
    # You can modify these parameters or pass them as command-line arguments
    
    # For local Qdrant (e.g., running in Docker)
    # qdrant_url = "http://localhost:6333"
    # qdrant_api_key = None
    
    # For Qdrant Cloud
    # qdrant_url = "https://your-qdrant-cloud-instance.qdrant.tech"
    # qdrant_api_key = "your-api-key-here"
    
    # Default to in-memory for testing
    qdrant_url = None
    qdrant_api_key = None
    
    # Update all implementations
    update_whisper_implementation(qdrant_url, qdrant_api_key)
    update_wave2vec_implementation(qdrant_url, qdrant_api_key)
    update_clap_implementation(qdrant_url, qdrant_api_key) 