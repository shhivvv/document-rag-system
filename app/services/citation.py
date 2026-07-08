import re
import logging
from typing import List, Dict, Any, Set
from app.schemas import Citation
from app.models import DocumentChunk

logger = logging.getLogger("rag_system.citation")

class CitationService:
    """
    Parses LLM responses for citation markers (e.g., [1], [2]) and maps them
    back to the original retrieved document chunks to construct formal citations.
    """

    @staticmethod
    def extract_citations(answer: str, retrieved_chunks: List[DocumentChunk]) -> List[Citation]:
        """
        Parses text answer looking for inline citations like [1], [2], etc.,
        and returns validated Citation objects with source files, pages, chunk IDs,
        and content snippets.
        """
        if not retrieved_chunks:
            return []

        # Find all brackets containing numbers, e.g., [1], [2]
        citation_matches = re.findall(r"\[(\d+)\]", answer)
        
        # Convert matches to 0-based indices and filter duplicates
        seen_indices: Set[int] = set()
        citations: List[Citation] = []

        for match in citation_matches:
            try:
                idx = int(match) - 1  # 1-indexed in LLM prompt, 0-indexed in list
                if 0 <= idx < len(retrieved_chunks):
                    seen_indices.add(idx)
                else:
                    logger.warning(f"LLM generated an out-of-bounds citation index: [{match}]")
            except ValueError:
                continue

        # In case the LLM forgot to write citations but did answer using context,
        # we can verify overlap or just return citations for chunks that have high similarity scores (e.g., > 0.8)
        # to ensure citations are never lost. But strict instructions make LLM cite.
        # Let's construct Citation objects for all cited chunks.
        for idx in sorted(seen_indices):
            chunk = retrieved_chunks[idx]
            
            # Extract citation details from chunk metadata
            metadata = chunk.metadata
            source_file = metadata.get("filename", "unknown")
            page = metadata.get("page_number")
            # If page is not in page_number, check 'page'
            if page is None:
                page = metadata.get("page")
                
            chunk_id = chunk.id
            snippet = chunk.text

            # Create Citation schema
            citations.append(
                Citation(
                    source_file=source_file,
                    page=int(page) if page is not None else None,
                    chunk_id=chunk_id,
                    snippet=snippet,
                    similarity_score=chunk.score
                )
            )

        logger.info(f"Successfully extracted {len(citations)} citations from response.")
        return citations

    @staticmethod
    def postprocess_citations_in_text(answer: str, citations: List[Citation], retrieved_chunks: List[DocumentChunk]) -> str:
        """
        Cleans up and verifies citation numbers. If the LLM generates a citation
        index that wasn't validated, we remove it from the text.
        """
        # Regex to find [N]
        pattern = re.compile(r"\[(\d+)\]")
        
        # Create mapping of old 1-based index in retrieved_chunks to valid citations indices (if any)
        # Wait, if we keep the numbers as they are but remove invalid ones:
        valid_indices = {i + 1 for i in range(len(retrieved_chunks))}
        
        def replace_match(match):
            val = int(match.group(1))
            if val in valid_indices:
                return f"[{val}]"
            return "" # Remove invalid citation tag

        cleaned_answer = pattern.sub(replace_match, answer)
        return cleaned_answer
