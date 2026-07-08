import json
import logging
from typing import Dict, Any
from app.services.retriever import DocumentRetriever
from app.services.llm import LLMService
from app.schemas import ContradictionResponse

logger = logging.getLogger("rag_system.contradiction")

class ContradictionService:
    """
    Analyzes semantic contradictions between two documents on a specified topic.
    Retrieves evidence from both documents and uses Groq to classify and detail conflicts.
    """

    def __init__(self, retriever: DocumentRetriever, llm_service: LLMService):
        self.retriever = retriever
        self.llm_service = llm_service

    def analyze_contradiction(self, doc1_name: str, doc2_name: str, topic: str) -> ContradictionResponse:
        """
        Retrieves relevant context for the topic from both documents,
        and uses the LLM to determine if there is a conflict.
        """
        logger.info(f"Analyzing contradiction between '{doc1_name}' and '{doc2_name}' on topic: '{topic}'")

        # 1. Retrieve chunks from Document 1
        chunks_doc1 = self.retriever.retrieve(query=topic, top_k=5, filename_filter=doc1_name)
        # 2. Retrieve chunks from Document 2
        chunks_doc2 = self.retriever.retrieve(query=topic, top_k=5, filename_filter=doc2_name)

        if not chunks_doc1:
            raise ValueError(f"No relevant content found in '{doc1_name}' for topic '{topic}'")
        if not chunks_doc2:
            raise ValueError(f"No relevant content found in '{doc2_name}' for topic '{topic}'")

        # Combine text segments
        evidence_doc1 = "\n\n".join([f"[Source: {c.metadata.get('filename')}, Page: {c.metadata.get('page_number')}]\n{c.text}" for c in chunks_doc1])
        evidence_doc2 = "\n\n".join([f"[Source: {c.metadata.get('filename')}, Page: {c.metadata.get('page_number')}]\n{c.text}" for c in chunks_doc2])

        # 3. System Prompt for Comparison
        system_prompt = (
            "You are an expert policy auditor and text analytics system. "
            "Your task is to compare the text from Document 1 and Document 2 concerning a specific topic. "
            "Determine if there is a 'Conflict', 'No Conflict', or 'Partial Conflict' between the two documents. "
            "\n"
            "Definitions:\n"
            "- Conflict: The documents make directly opposing or incompatible claims on the topic.\n"
            "- No Conflict: The documents are completely aligned, or one document contains details that supplement the other without contradiction.\n"
            "- Partial Conflict: The documents are mostly aligned, but contain minor discrepancies, varying details, or differing exceptions.\n"
            "\n"
            "You MUST return your output as a single, valid JSON object with the following keys:\n"
            "- status: exactly 'Conflict', 'No Conflict', or 'Partial Conflict'\n"
            "- reasoning: a detailed explanation of why you made this classification, explaining the differences or alignment.\n"
            "- document1_evidence: direct quotes or summaries of evidence from Document 1 supporting your conclusion.\n"
            "- document2_evidence: direct quotes or summaries of evidence from Document 2 supporting your conclusion.\n"
            "- confidence: a confidence float score between 0.0 and 1.0 (e.g. 0.95).\n"
            "\n"
            "Do NOT include any markdown blocks (like ```json), commentary, introduction, or text outside the JSON object."
        )

        prompt = (
            f"Topic: {topic}\n\n"
            f"=== DOCUMENT 1 TEXT ===\n{evidence_doc1}\n\n"
            f"=== DOCUMENT 2 TEXT ===\n{evidence_doc2}\n\n"
            f"Compare and analyze. Return only the JSON response."
        )

        try:
            # Query LLM with JSON format expectation
            response_raw = self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"}
            ).strip()

            logger.debug(f"Raw contradiction LLM response: {response_raw}")
            
            # Parse the JSON response
            data = json.loads(response_raw)
            
            # Validate and clean fields
            status = data.get("status", "No Conflict")
            if status not in ["Conflict", "No Conflict", "Partial Conflict"]:
                status = "No Conflict"

            return ContradictionResponse(
                status=status,
                reasoning=data.get("reasoning", "No reasoning provided."),
                document1_evidence=data.get("document1_evidence", "Evidence from doc 1 not isolated."),
                document2_evidence=data.get("document2_evidence", "Evidence from doc 2 not isolated."),
                confidence=float(data.get("confidence", 0.5))
            )

        except Exception as e:
            logger.error(f"Error during contradiction LLM analysis: {str(e)}")
            # Fail gracefully by providing a standard error response or re-raising
            raise RuntimeError(f"Contradiction analysis execution failed: {str(e)}") from e
