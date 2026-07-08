import hashlib
import logging
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger("rag_system.embedding")

class EmbeddingService:
    """
    Service for generating vector embeddings from text strings.
    Uses SentenceTransformers (default: all-MiniLM-L6-v2) and implements an in-memory cache
    to avoid redundant calls for identical text chunks.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}") from e

        # In-memory cache: md5_hash(text) -> list[float]
        self._cache: Dict[str, List[float]] = {}

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        """
        text_hash = self._get_hash(text)
        if text_hash in self._cache:
            return self._cache[text_hash]

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            emb_list = embedding.tolist()
            self._cache[text_hash] = emb_list
            return emb_list
        except Exception as e:
            logger.error(f"Error encoding query text: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document chunks. Uses cache when available,
        batches missing ones, and updates the cache.
        """
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, text in enumerate(texts):
            text_hash = self._get_hash(text)
            if text_hash in self._cache:
                results[idx] = self._cache[text_hash]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            logger.info(f"Computing embeddings for {len(missing_texts)} uncached chunks.")
            try:
                embeddings = self.model.encode(missing_texts, show_progress_bar=False, convert_to_numpy=True)
                for idx, emb in zip(missing_indices, embeddings):
                    emb_list = emb.tolist()
                    text_hash = self._get_hash(texts[idx])
                    self._cache[text_hash] = emb_list
                    results[idx] = emb_list
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise RuntimeError(f"Batch embedding generation failed: {str(e)}") from e

        return results

    def clear_cache(self):
        """Clears the internal cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared.")
