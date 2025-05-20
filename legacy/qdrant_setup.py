import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Dict, Any, Optional, List


class QdrantConnector:
    """
    A connector class for managing connections to Qdrant and creating collections
    for different audio vectorization approaches.
    """
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, in_memory: bool = False):
        """
        Initialize the Qdrant connector.
        
        Args:
            url: URL of the Qdrant server
            api_key: API key for Qdrant Cloud
            in_memory: Whether to use an in-memory instance for testing
        """
        self.url = url
        self.api_key = api_key
        self.in_memory = in_memory
        
        # Initialize client based on parameters
        if in_memory:
            self.client = QdrantClient(":memory:")
            self.connection_type = "in-memory"
            print("Connected to in-memory Qdrant instance")
        elif url:
            if api_key:
                self.client = QdrantClient(url=url, api_key=api_key)
                self.connection_type = "cloud"
                print(f"Connected to Qdrant Cloud at {url}")
            else:
                self.client = QdrantClient(url=url)
                self.connection_type = "remote"
                print(f"Connected to Qdrant server at {url}")
        else:
            # Default to local instance
            self.client = QdrantClient(":memory:")
            self.connection_type = "in-memory"
            print("No URL provided. Connected to in-memory Qdrant instance")
    
    def create_whisper_collection(self, collection_name: str = "whisper_audio_collection") -> bool:
        """
        Create a collection for Whisper-based audio embeddings.
        OpenAI embeddings are 1536-dimensional.
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            bool: Whether the collection was created successfully
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(collection.name == collection_name for collection in collections):
                print(f"Collection '{collection_name}' already exists")
                return True
            
            # Create new collection
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embeddings dimension
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created collection '{collection_name}' for Whisper embeddings")
            return True
        except Exception as e:
            print(f"Error creating collection '{collection_name}': {str(e)}")
            return False
    
    def create_wav2vec_collection(self, collection_name: str = "wav2vec_audio_collection") -> bool:
        """
        Create a collection for Wav2Vec-based audio embeddings.
        Wav2Vec embeddings are 768-dimensional.
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            bool: Whether the collection was created successfully
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(collection.name == collection_name for collection in collections):
                print(f"Collection '{collection_name}' already exists")
                return True
            
            # Create new collection
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Wav2Vec embeddings dimension
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created collection '{collection_name}' for Wav2Vec embeddings")
            return True
        except Exception as e:
            print(f"Error creating collection '{collection_name}': {str(e)}")
            return False
    
    def create_clap_collection(self, collection_name: str = "clap_audio_collection") -> bool:
        """
        Create a collection for CLAP-based audio embeddings.
        CLAP embeddings are 512-dimensional.
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            bool: Whether the collection was created successfully
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(collection.name == collection_name for collection in collections):
                print(f"Collection '{collection_name}' already exists")
                return True
            
            # Create new collection
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=512,  # CLAP embeddings dimension
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Created collection '{collection_name}' for CLAP embeddings")
            return True
        except Exception as e:
            print(f"Error creating collection '{collection_name}': {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict: Collection information
        """
        try:
            return self.client.get_collection(collection_name=collection_name).dict()
        except Exception as e:
            print(f"Error getting collection '{collection_name}': {str(e)}")
            return {}
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the Qdrant instance.
        
        Returns:
            List[str]: List of collection names
        """
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: Whether the collection was deleted successfully
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Choose one of these connection methods
    
    # 1. In-memory Qdrant instance (for testing, data is lost when the process ends)
    connector_memory = QdrantConnector(in_memory=True)
    
    # 2. Local persistent Qdrant instance (requires Qdrant to be installed locally)
    # You can install Qdrant locally using Docker:
    # docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    connector_local = QdrantConnector(url="http://localhost:6333")
    
    # 3. Qdrant Cloud instance (for production use)
    # Sign up at https://cloud.qdrant.io/ to get a URL and API key
    # connector_cloud = QdrantConnector(
    #     url="https://your-qdrant-cloud-instance.qdrant.tech",
    #     api_key="your-api-key-here"
    # )
    
    # For this example, we'll use the in-memory instance
    connector = connector_memory
    
    # Create collections for each approach
    connector.create_whisper_collection("whisper_collection")
    connector.create_wav2vec_collection("wav2vec_collection")
    connector.create_clap_collection("clap_collection")
    
    # List collections
    collections = connector.list_collections()
    print(f"Available collections: {collections}")
    
    # Get information about a specific collection
    if collections:
        collection_info = connector.get_collection_info(collections[0])
        print(f"\nCollection info for '{collections[0]}':")
        print(f"  Vector size: {collection_info['config']['params']['vectors']['size']}")
        print(f"  Distance: {collection_info['config']['params']['vectors']['distance']}")
        print(f"  Points count: {collection_info['points_count']}")
    
    print("\nTo use persistent storage with your audio vectorization scripts:")
    print("1. Initialize QdrantConnector with appropriate URL/API key")
    print("2. Use the same collection names across sessions")
    print("3. Update the scripts to use the connector with persistent storage")