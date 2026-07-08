import time
import logging
from typing import List, Dict, Any, Tuple
from app.services.llm import LLMService
from app.services.retriever import DocumentRetriever
from app.services.citation import CitationService
from app.services.translation import TranslationService
from app.schemas import AskRequest, AskResponse, Citation

logger = logging.getLogger("rag_system.rag_pipeline")

class RAGPipeline:
    """
    Core RAG orchestrator that manages:
    - Language detection and query translation (multilingual support)
    - Vector search retrieval (similarity or MMR)
    - Strict context-bound question answering
    - Citation extraction
    - Answer translation and final response schema mapping
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever: DocumentRetriever,
        citation_service: CitationService,
        translation_service: TranslationService
    ):
        self.llm_service = llm_service
        self.retriever = retriever
        self.citation_service = citation_service
        self.translation_service = translation_service

    def _build_context_string(self, chunks: List[Any]) -> str:
        """Formats the retrieved document chunks into a structured string for the prompt."""
        context_parts = []
        for idx, chunk in enumerate(chunks):
            # 1-indexed for citation mapping
            ref_num = idx + 1
            source_info = f"Source {ref_num} - File: {chunk.metadata.get('filename')}, Page: {chunk.metadata.get('page_number')}, Chunk ID: {chunk.id}"
            context_parts.append(f"[{ref_num}] ({source_info})\nContent: {chunk.text}")
        return "\n\n".join(context_parts)

    def answer_question(self, request: AskRequest) -> AskResponse:
        """
        Orchestrates language detection, translation, context retrieval, context-bound answering,
        citation alignment, and final language formatting.
        Tracks response latency.
        """
        start_time = time.time()
        original_query = request.question.strip()
        logger.info(f"RAG Pipeline started for question: '{original_query[:100]}...'")

        # 1. Detect language
        target_lang = request.target_language
        # If target_lang is the Swagger default placeholder "string" or empty, ignore it
        if target_lang and (target_lang.strip().lower() == "string" or not target_lang.strip()):
            target_lang = None

        if target_lang:
            # Normalize target language name
            lang_lower = target_lang.strip().lower()
            if lang_lower in self.translation_service.SUPPORTED_LANGUAGES:
                detected_lang = self.translation_service.SUPPORTED_LANGUAGES[lang_lower]
            else:
                detected_lang = self.translation_service.detect_language(original_query)
        else:
            detected_lang = self.translation_service.detect_language(original_query)
        
        # 2. Translate question to English if necessary
        english_query = original_query
        if detected_lang.lower() != "english":
            logger.info(f"Translating query from {detected_lang} to English.")
            english_query = self.translation_service.translate_to_english(original_query, detected_lang)

        # 3. Retrieve chunks
        # Use Top K = 5 as required
        retrieved_chunks = self.retriever.retrieve(
            query=english_query,
            top_k=5,
            use_mmr=request.use_mmr
        )

        if not retrieved_chunks:
            logger.warning("No chunks retrieved from vector database.")
            latency = time.time() - start_time
            return AskResponse(
                question=original_query,
                answer="I could not find this information in the provided documents.",
                language=detected_lang,
                citations=[],
                latency_seconds=round(latency, 3)
            )

        # 4. Format context for prompt
        context_str = self._build_context_string(retrieved_chunks)

        # 5. QA Generation
        system_prompt = (
            "You are a precise, strict, fact-based document QA assistant.\n"
            "Your task is to answer the user's question using ONLY the retrieved context fragments provided below.\n\n"
            "STRICT RULES:\n"
            "1. Base your answer ONLY on the provided context fragments. Do NOT extrapolate, guess, or use external knowledge.\n"
            "2. If the context does not contain the answer, you MUST respond EXACTLY: 'I could not find this information in the provided documents.' and nothing else.\n"
            "3. For every fact or claim you make, you must cite the source by appending the index of the matching context fragment in brackets, e.g. [1], [2], [1][2], etc.\n"
            "4. Never hallucinate facts, dates, numbers, or names. If they are not in the context, they do not exist.\n"
            "5. Generate the answer in English."
        )

        prompt = (
            f"Retrieved Context Fragments:\n"
            f"-----------------------------------------\n"
            f"{context_str}\n"
            f"-----------------------------------------\n\n"
            f"Question: {english_query}\n\n"
            f"Provide a clear and factual answer following the rules above:"
        )

        try:
            logger.info("Generating answer from Groq LLM...")
            english_answer = self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=1000
            ).strip()
            
            logger.debug(f"Raw English answer: {english_answer}")
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            latency = time.time() - start_time
            return AskResponse(
                question=original_query,
                answer="Failed to generate answer due to model timeout or connection issue.",
                language=detected_lang,
                citations=[],
                latency_seconds=round(latency, 3)
            )

        # Check if the LLM emitted the "not found" response
        not_found_phrase = "I could not find this information in the provided documents."
        if not_found_phrase.lower() in english_answer.lower():
            logger.info("LLM determined that information is unavailable in context.")
            latency = time.time() - start_time
            
            # If the user asked in a non-English language, translate the "not found" response back
            final_answer = not_found_phrase
            if detected_lang.lower() != "english":
                final_answer = self.translation_service.translate_from_english(not_found_phrase, detected_lang)

            return AskResponse(
                question=original_query,
                answer=final_answer,
                language=detected_lang,
                citations=[],
                latency_seconds=round(latency, 3)
            )

        # 6. Extract citations from English answer
        citations = self.citation_service.extract_citations(english_answer, retrieved_chunks)
        
        # Clean up any invalid citation markers in the English text
        english_answer = self.citation_service.postprocess_citations_in_text(english_answer, citations, retrieved_chunks)

        # 7. Translate answer back to original query language
        final_answer = english_answer
        if detected_lang.lower() != "english":
            logger.info(f"Translating answer back to original language: {detected_lang}")
            final_answer = self.translation_service.translate_from_english(english_answer, detected_lang)

        latency = time.time() - start_time
        logger.info(f"RAG Pipeline finished in {latency:.3f} seconds.")

        return AskResponse(
            question=original_query,
            answer=final_answer,
            language=detected_lang,
            citations=citations,
            latency_seconds=round(latency, 3)
        )
