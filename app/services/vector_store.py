import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.models import DocumentChunk
from app.services.embedding import EmbeddingService

logger = logging.getLogger("rag_system.vector_store")

class VectorStoreService:
    """
    Service wrapper around ChromaDB persistent vector database.
    Manages indexing, duplicate prevention, collection clearing, and querying.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.persist_dir = str(settings.get_chroma_path())
        logger.info(f"Connecting to ChromaDB persisted at: {self.persist_dir}")
        
        # Configure Chroma persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "document_rag_collection"
        self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity for cosine distance query calculations
            )
            logger.info(f"Connected to ChromaDB collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error connecting to Chroma collection: {str(e)}")
            raise RuntimeError(f"ChromaDB connection failed: {str(e)}") from e

    def reload_collection(self):
        """Reloads the collection by re-fetching it from the client."""
        logger.info("Reloading ChromaDB collection.")
        self._get_or_create_collection()

    def delete_collection(self):
        """Deletes the current collection and recreates it empty."""
        logger.warning(f"Deleting ChromaDB collection: '{self.collection_name}'")
        try:
            self.client.delete_collection(name=self.collection_name)
            self._get_or_create_collection()
            logger.info("ChromaDB collection recreated and cleared successfully.")
        except Exception as e:
            logger.error(f"Error resetting vector collection: {str(e)}")
            # Fallback recreate
            self._get_or_create_collection()

    def add_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Inserts document chunks into Chroma DB.
        Performs duplicate detection by checking if chunk IDs already exist in the store.
        Uses cached embeddings.
        """
        if not chunks:
            return {"added": 0, "skipped": 0}

        # Check existing IDs in Chroma to prevent duplicate ingestion
        chunk_ids = [c.id for c in chunks]
        
        try:
            existing_results = self.collection.get(ids=chunk_ids)
            existing_ids = set(existing_results.get("ids", []))
        except Exception as e:
            logger.warning(f"Could not retrieve existing IDs for duplicate check: {str(e)}. Proceeding with full insertion.")
            existing_ids = set()

        new_chunks = [c for c in chunks if c.id not in existing_ids]
        skipped_count = len(chunks) - len(new_chunks)

        if not new_chunks:
            logger.info("All chunks already exist in vector store. Skipping ingestion.")
            return {"added": 0, "skipped": skipped_count}

        # Embed texts of new chunks
        texts_to_embed = [c.text for c in new_chunks]
        embeddings = self.embedding_service.embed_documents(texts_to_embed)

        ids = [c.id for c in new_chunks]
        metadatas = [c.metadata for c in new_chunks]
        
        # Chroma expects plain metadata types (string, int, float, bool)
        # Convert any non-supported metadata fields just in case
        sanitized_metadatas = []
        for meta in metadatas:
            sanitized = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                elif v is None:
                    sanitized[k] = ""
                else:
                    sanitized[k] = str(v)
            sanitized_metadatas.append(sanitized)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=sanitized_metadatas,
                documents=texts_to_embed
            )
            logger.info(f"Ingested {len(new_chunks)} new chunks. Skipped {skipped_count} duplicates.")
            return {"added": len(new_chunks), "skipped": skipped_count}
        except Exception as e:
            logger.error(f"Error adding chunks to Chroma collection: {str(e)}")
            raise RuntimeError(f"ChromaDB write failed: {str(e)}") from e

    def get_document_count(self) -> int:
        """Returns the number of unique documents indexed."""
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            doc_ids = set()
            for meta in metadatas:
                if meta and "document_id" in meta:
                    doc_ids.add(meta["document_id"])
            return len(doc_ids)
        except Exception as e:
            logger.error(f"Failed to count documents in vector store: {str(e)}")
            return 0

    def get_chunk_count(self) -> int:
        """Returns the total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count chunks in vector store: {str(e)}")
            return 0

    def get_all_document_names(self) -> List[str]:
        """Returns list of all source filenames currently indexed in the vector store."""
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            filenames = set()
            for meta in metadatas:
                if meta and "filename" in meta:
                    filenames.add(meta["filename"])
            return list(filenames)
        except Exception as e:
            logger.error(f"Failed to retrieve document names: {str(e)}")
            return []

    def query(self, query_vector: List[float], top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Executes query on vector store. Returns DocumentChunks containing
        the texts, metadata and distance calculation converted to similarity scores.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            raise RuntimeError(f"Vector search failed: {str(e)}") from e

        chunks = []
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # Convert cosine distance to cosine similarity score: 1 - distance
            dist = distances[i]
            similarity_score = float(round(1.0 - dist, 4))
            
            chunk = DocumentChunk(
                id=ids[i],
                document_id=metadatas[i].get("document_id", ""),
                text=documents[i],
                metadata=metadatas[i],
                score=similarity_score
            )
            chunks.append(chunk)

        return chunks
