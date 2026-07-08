import logging
from typing import List, Dict, Any, Optional
import numpy as np
from app.models import DocumentChunk
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger("rag_system.retriever")

class DocumentRetriever:
    """
    Service layer coordinating vector store queries.
    Supports standard similarity search and optional Maximal Marginal Relevance (MMR)
    for diversity in retrieval.
    """

    def __init__(self, vector_store: VectorStoreService, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _maximal_marginal_relevance(
        self,
        query_vector: np.ndarray,
        candidate_embeddings: List[List[float]],
        candidate_chunks: List[DocumentChunk],
        top_k: int,
        lambda_val: float = 0.5
    ) -> List[DocumentChunk]:
        """
        Implementation of MMR selection algorithm.
        Balances query relevance and diversity among selected documents.
        """
        if not candidate_chunks:
            return []
            
        selected_indices: List[int] = []
        candidates_left = list(range(len(candidate_chunks)))
        
        # Convert candidate embeddings list to numpy array
        candidate_embs = [np.array(emb) for emb in candidate_embeddings]
        q_vec = np.array(query_vector)

        # Loop until top_k chunks are selected or we run out of candidates
        for _ in range(min(top_k, len(candidate_chunks))):
            best_mmr = -float("inf")
            best_idx = -1

            for idx in candidates_left:
                sim_to_query = self._cosine_similarity(candidate_embs[idx], q_vec)
                
                # Compute maximum similarity to already selected chunks
                if not selected_indices:
                    max_sim_to_selected = 0.0
                else:
                    max_sim_to_selected = max(
                        self._cosine_similarity(candidate_embs[idx], candidate_embs[sel_idx])
                        for sel_idx in selected_indices
                    )
                
                # MMR formula
                mmr_score = lambda_val * sim_to_query - (1 - lambda_val) * max_sim_to_selected
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            candidates_left.remove(best_idx)

        return [candidate_chunks[idx] for idx in selected_indices]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = False,
        filename_filter: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve chunks matching query.
        If use_mmr is True, queries double the candidates and filters using MMR.
        If filename_filter is provided, restricts search to that document.
        """
        logger.info(f"Retrieving for query: '{query}' (top_k={top_k}, mmr={use_mmr}, filter={filename_filter})")
        
        # Generate query vector
        query_vector = self.embedding_service.embed_query(query)
        
        # Prepare filter condition for Chroma
        where_clause = None
        if filename_filter:
            where_clause = {"filename": filename_filter}

        if not use_mmr:
            # Standard Similarity Search
            return self.vector_store.query(query_vector, top_k=top_k, where=where_clause)
        
        # MMR Search: fetch candidate pool first (e.g. top_k * 3)
        candidate_pool_size = max(top_k * 3, 15)
        try:
            # Query collection including embeddings to perform MMR calculation
            results = self.vector_store.collection.query(
                query_embeddings=[query_vector],
                n_results=candidate_pool_size,
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
        except Exception as e:
            logger.error(f"ChromaDB candidate fetch for MMR failed: {str(e)}")
            # Fallback to standard query
            return self.vector_store.query(query_vector, top_k=top_k, where=where_clause)

        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = results["embeddings"][0]

        candidate_chunks = []
        for i in range(len(ids)):
            dist = distances[i]
            similarity_score = float(round(1.0 - dist, 4))
            
            chunk = DocumentChunk(
                id=ids[i],
                document_id=metadatas[i].get("document_id", ""),
                text=documents[i],
                metadata=metadatas[i],
                score=similarity_score
            )
            candidate_chunks.append(chunk)

        # Apply MMR re-ranking
        selected_chunks = self._maximal_marginal_relevance(
            query_vector=query_vector,
            candidate_embeddings=embeddings,
            candidate_chunks=candidate_chunks,
            top_k=top_k
        )
        
        return selected_chunks
