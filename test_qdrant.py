import os
import argparse
import uuid
from qdrant_setup import QdrantConnector
from typing import Optional
from qdrant_client.http import models


def test_qdrant_connection(url: Optional[str] = None, api_key: Optional[str] = None, in_memory: bool = False):
    """
    Test connection to Qdrant and basic operations
    
    Args:
        url: URL of the Qdrant server
        api_key: API key for Qdrant Cloud
        in_memory: Whether to use in-memory instance
    """
    print("\n=== Testing Qdrant Connection ===")
    
    # Create connector
    connector = QdrantConnector(url=url, api_key=api_key, in_memory=in_memory)
    
    # Test creating collections
    print("\nCreating collections...")
    whisper_result = connector.create_whisper_collection("test_whisper")
    wav2vec_result = connector.create_wav2vec_collection("test_wav2vec")
    clap_result = connector.create_clap_collection("test_clap")
    
    # List collections
    print("\nListing collections...")
    collections = connector.list_collections()
    print(f"Available collections: {collections}")
    
    # Get collection info
    print("\nGetting collection info...")
    for collection in collections:
        info = connector.get_collection_info(collection)
        vector_size = info.get('config', {}).get('params', {}).get('vectors', {}).get('size')
        print(f"Collection '{collection}' - Vector size: {vector_size}")
    
    # Test adding points to a collection
    print("\nTesting adding points to collection...")
    client = connector.client
    
    # We'll use proper UUIDs for point IDs
    point_id_1 = str(uuid.uuid4())
    point_id_2 = str(uuid.uuid4())
    point_id_3 = str(uuid.uuid4())
    
    try:
        # Add to whisper collection (1536 dimensions)
        whisper_vector = [0.1] * 1536
        client.upsert(
            collection_name="test_whisper",
            points=[
                models.PointStruct(
                    id=point_id_1,
                    vector=whisper_vector,
                    payload={"text": "This is a test point for Whisper"}
                )
            ]
        )
        print("Successfully added point to test_whisper collection")
        
        # Add to wav2vec collection (768 dimensions)
        wav2vec_vector = [0.1] * 768
        client.upsert(
            collection_name="test_wav2vec",
            points=[
                models.PointStruct(
                    id=point_id_2,
                    vector=wav2vec_vector,
                    payload={"text": "This is a test point for Wav2Vec"}
                )
            ]
        )
        print("Successfully added point to test_wav2vec collection")
        
        # Add to CLAP collection (512 dimensions)
        clap_vector = [0.1] * 512
        client.upsert(
            collection_name="test_clap",
            points=[
                models.PointStruct(
                    id=point_id_3,
                    vector=clap_vector,
                    payload={"text": "This is a test point for CLAP"}
                )
            ]
        )
        print("Successfully added point to test_clap collection")
        
        # Test search
        print("\nTesting search in collections...")
        for collection_name in ["test_whisper", "test_wav2vec", "test_clap"]:
            dimensions = {"test_whisper": 1536, "test_wav2vec": 768, "test_clap": 512}
            query_vector = [0.1] * dimensions[collection_name]
            
            # Use search method with correct parameters
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=1
            )
            
            # Extract payloads safely
            result_payloads = []
            for result in results:
                if hasattr(result, "payload"):
                    result_payloads.append(result.payload)
                elif isinstance(result, dict) and "payload" in result:
                    result_payloads.append(result["payload"])
            
            print(f"Search results for {collection_name}: {result_payloads}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Clean up test collections
    print("\nCleaning up test collections...")
    for collection in ["test_whisper", "test_wav2vec", "test_clap"]:
        connector.delete_collection(collection)
    
    print("\nTest completed")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Qdrant connection")
    parser.add_argument("--url", help="URL of the Qdrant server")
    parser.add_argument("--api-key", help="API key for Qdrant Cloud")
    parser.add_argument("--in-memory", action="store_true", help="Use in-memory instance")
    args = parser.parse_args()
    
    # Use environment variables as fallback
    url = args.url or os.environ.get("QDRANT_URL")
    api_key = args.api_key or os.environ.get("QDRANT_API_KEY")
    in_memory = args.in_memory or not (url or api_key)
    
    # Run test
    test_qdrant_connection(url=url, api_key=api_key, in_memory=in_memory) 