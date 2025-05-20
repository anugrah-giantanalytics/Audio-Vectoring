import os
from typing import Optional, List, Dict, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

class QdrantConnector:
    """
    A connector class for Qdrant vector database operations.
    """
    def __init__(
        self,
        collection_name: str,
        vector_size: int = 1536,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        in_memory: bool = False,
        path: Optional[str] = None
    ):
        """
        Initialize the Qdrant connector.
        
        Args:
            collection_name: Name of the collection to use
            vector_size: Size of the embedding vectors
            url: URL of the Qdrant server (if using a remote server)
            api_key: API key for Qdrant Cloud (if applicable)
            in_memory: Whether to use in-memory storage
            path: Path for local storage (if not using in-memory or remote)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Client initialization based on parameters
        if url:
            # Remote server
            self.client = QdrantClient(url=url, api_key=api_key)
        elif in_memory:
            # In-memory storage
            self.client = QdrantClient(":memory:")
        elif path:
            # Local storage
            self.client = QdrantClient(path=path)
        else:
            # Default to in-memory if no other options provided
            self.client = QdrantClient(":memory:")
        
        # Ensure collection exists
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist yet."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        """
        Add documents and their embeddings to the collection.
        
        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
            ids: Optional list of IDs for the points
            batch_size: Number of vectors to upload in one batch
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Prepare points for insertion
        points = []
        for i in range(len(documents)):
            points.append(
                models.PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload={
                        "page_content": documents[i].page_content,
                        "metadata": documents[i].metadata
                    }
                )
            )
        
        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        return ids
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None
    ):
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            filter_condition: Optional filter to apply to the search
        
        Returns:
            List of search results with documents and scores
        """
        filter_param = None
        if filter_condition:
            filter_param = models.Filter(**filter_condition)
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_param
        )
        
        results = []
        for res in search_result:
            # Convert Qdrant point back to Document
            document = Document(
                page_content=res.payload.get("page_content", ""),
                metadata=res.payload.get("metadata", {})
            )
            
            results.append({
                "document": document,
                "score": res.score,
                "id": res.id
            })
        
        return results
    
    def delete(self, ids: List[str]):
        """
        Delete points from the collection by IDs.
        
        Args:
            ids: List of point IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )
    
    def clear_collection(self):
        """Delete all points from the collection."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter()
            )
        )

    def create_collection(self, collection_name: str, vector_size: int, 
                         distance: Distance = Distance.COSINE,
                         sparse: bool = False) -> None:
        """
        Create a new collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension
            distance: Distance metric to use
            sparse: Whether to enable sparse vectors for hybrid search
        """
        # Check if collection already exists
        try:
            self.client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists")
            return
        except Exception:
            # Collection doesn't exist, create it
            pass
        
        vectors_config = VectorParams(
            size=vector_size,
            distance=distance,
        )
        
        # Create collection
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
        
        # Enable sparse vectors if requested
        if sparse:
            sparse_vector_config = VectorParams(
                size=vector_size,
                distance=Distance.DOT,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="sparse_vector",
                field_schema=sparse_vector_config,
            )
        
        print(f"Created collection '{collection_name}' for vectors of size {vector_size}")
        
    def create_whisper_collection(self, collection_name: str) -> None:
        """Create a collection for Whisper embeddings (OpenAI)"""
        self.create_collection(collection_name, vector_size=1536, sparse=True)
        
    def create_wav2vec_collection(self, collection_name: str) -> None:
        """Create a collection for Wav2Vec embeddings"""
        self.create_collection(collection_name, vector_size=768)
        
    def create_clap_collection(self, collection_name: str) -> None:
        """Create a collection for CLAP embeddings"""
        self.create_collection(collection_name, vector_size=512) 