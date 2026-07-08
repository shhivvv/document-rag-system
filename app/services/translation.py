import logging
from typing import Dict, Tuple, Optional
from app.services.llm import LLMService

logger = logging.getLogger("rag_system.translation")

class TranslationService:
    """
    Isolated translation service using Groq API.
    Detects query language and translates queries to English and answers back to original languages.
    """

    SUPPORTED_LANGUAGES = {
        "english": "English",
        "hindi": "Hindi",
        "french": "French",
        "spanish": "Spanish",
        "german": "German",
        "japanese": "Japanese",
        "marathi": "Marathi"
    }

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text using Groq LLM.
        Returns one of the supported languages. Default is English.
        """
        if not text or not text.strip():
            return "English"

        system_prompt = (
            "You are an expert language identifier. Your task is to detect the language of the user's text. "
            "You must output exactly one language name from this list: English, Hindi, French, Spanish, German, Japanese, Marathi. "
            "If the language is not in this list or you are unsure, output 'English'. "
            "Do NOT write any introduction, explanation, punctuation, or extra words. Output ONLY the single language name."
        )
        
        prompt = f"Identify the language of the following text:\n\n{text}"
        
        try:
            detected = self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=10
            ).strip().lower()
            
            # Clean up the output in case LLM added periods or extra whitespace
            for lang_key, lang_name in self.SUPPORTED_LANGUAGES.items():
                if lang_key in detected:
                    logger.info(f"Language detected: {lang_name}")
                    return lang_name
                    
            logger.info("Language not match supported list. Defaulting to English.")
            return "English"
        except Exception as e:
            logger.error(f"Error during language detection: {str(e)}. Defaulting to English.")
            return "English"

    def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translates text from source_language to English using Groq LLM.
        """
        if source_language.lower() == "english":
            return text

        system_prompt = (
            f"You are an expert translator translating text from {source_language} to English. "
            "Translate the user text accurately while preserving technical terms and semantic meaning. "
            "Output ONLY the translated text. Do NOT add any preamble, conversational filler, or commentary."
        )
        
        prompt = f"Translate the following text to English:\n\n{text}"
        
        try:
            translation = self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=1000
            ).strip()
            logger.debug(f"Translated query to English: {translation}")
            return translation
        except Exception as e:
            logger.error(f"Failed to translate query to English: {str(e)}")
            raise RuntimeError(f"Translation to English failed: {str(e)}") from e

    def translate_from_english(self, text: str, target_language: str) -> str:
        """
        Translates text from English to target_language using Groq LLM.
        Preserves citation tags like [1] or [2] exactly where they are in the source text.
        """
        if target_language.lower() == "english":
            return text

        system_prompt = (
            f"You are an expert translator translating text from English to {target_language}. "
            "IMPORTANT: Preserve all bracketed numbers (e.g. [1], [2], [10]) exactly as they appear in the original text. "
            "Do not translate or remove these citation markers, and place them at the matching semantic positions in the translated output. "
            "Output ONLY the translated text. Do NOT add any notes, headers, or explanations."
        )
        
        prompt = f"Translate the following English text into {target_language}, preserving [num] markers:\n\n{text}"
        
        try:
            translation = self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=2000
            ).strip()
            logger.debug(f"Translated answer back to {target_language}")
            return translation
        except Exception as e:
            logger.error(f"Failed to translate answer from English to {target_language}: {str(e)}")
            # Fall back to English text if translation fails, rather than failing completely
            return f"{text} (Translation failed)"
ClassTranslationService = TranslationService
