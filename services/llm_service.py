"""
services/llm_service.py

Google Gemini chat-model service for HormoneBench AI.

Compatible with:
- Python 3.11+
- LangChain 1.3.x
- langchain-core 1.4.x
- langchain-google-genai 4.x
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


if not (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
):
    load_dotenv(BASE_DIR / ".env.example", override=False)


class LLMService:
    """
    Initialize and return a Google Gemini chat model.

    Required environment variable:

        GOOGLE_API_KEY=your_api_key

    Fallback environment variable:

        GEMINI_API_KEY=your_api_key

    Optional environment variable:

        GEMINI_MODEL=gemini-2.5-flash
    """

    def __init__(self) -> None:
        """
        Read Gemini configuration from environment variables.
        """

        self.api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Missing Google Gemini API key. "
                "Set GOOGLE_API_KEY in your .env file or use "
                "GEMINI_API_KEY as a fallback. "
                "If you are using the sample config, copy "
                ".env.example to .env."
            )

        self.model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

        self.temperature = 0.2
        self.max_tokens = 1500
        self.timeout = 120
        self.max_retries = 2

    def get_llm(
        self,
    ) -> ChatGoogleGenerativeAI:
        """
        Create and return the configured Gemini chat model.
        """

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
            vertexai=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def health(self) -> dict[str, object]:
        """
        Return non-sensitive LLM configuration.
        """

        return {
            "provider": "Google Gemini",
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }