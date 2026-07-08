import os
import logging
from typing import Optional, List, Dict, Any
from groq import Groq
from app.config import settings

logger = logging.getLogger("rag_system.llm")

class LLMService:
    """
    Wrapper service for interacting with the Groq API.
    Provides methods for generating text completions with custom prompts and handling API connection issues.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_MODEL
        self.client = None

        if not self.api_key:
            logger.warning("GROQ_API_KEY is not configured. LLM calls will fail until the key is set.")
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq client initialized using model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")

    def is_configured(self) -> bool:
        """Returns True if the Groq client is successfully configured."""
        return self.client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Sends generation request to Groq LLM endpoint.
        Uses a temperature of 0.0 by default to keep answers deterministic and factual.
        """
        if not self.client:
            # Try lazy re-initialization if a key was added to environment variables
            self.api_key = self.api_key or os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
            if self.api_key:
                try:
                    self.client = Groq(api_key=self.api_key)
                except Exception as e:
                    logger.error(f"Lazy Groq initialization failed: {str(e)}")
            
            if not self.client:
                raise ValueError(
                    "Groq API client is not configured. Please set the GROQ_API_KEY environment variable."
                )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Sending request to Groq LLM ({self.model_name}) with temperature {temperature}")
            
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            chat_completion = self.client.chat.completions.create(**kwargs)
            response_text = chat_completion.choices[0].message.content
            if not response_text:
                raise ValueError("Received empty response content from Groq LLM")
            return response_text
            
        except Exception as e:
            logger.error(f"Error during Groq API execution: {str(e)}")
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e
