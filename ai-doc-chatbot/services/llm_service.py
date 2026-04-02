"""
llm_service.py
--------------
Sends the retrieved context + user query to an LLM and returns the answer.

Supports three providers (set LLM_PROVIDER in .env):
  - groq    → fastest, free tier, uses Llama 3
  - openai  → GPT-4o
  - gemini  → Google Gemini

The system prompt instructs the LLM to ONLY answer from the provided context.
This prevents hallucination — it won't invent answers from general knowledge.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    """Handles all communication with the chosen LLM provider."""

    SYSTEM_PROMPT = """You are a helpful document assistant.
Answer the user's question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I couldn't find that information in the uploaded document."
Do not use outside knowledge. Be concise and accurate.
"""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()
        self._client = None
        print(f"[LLMService] Using provider: {self.provider}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def answer(self, query: str, context: str, chat_history: list = None) -> str:
        """
        Send query + context to the LLM and return the answer string.

        Args:
            query:        The user's question.
            context:      Retrieved document excerpts (from RetrievalService).
            chat_history: Optional list of past {"role": ..., "content": ...} messages.
        """
        messages = self._build_messages(query, context, chat_history or [])

        if self.provider == "groq":
            return self._call_groq(messages)
        elif self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "gemini":
            return self._call_gemini(query, context)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{self.provider}'. Choose groq, openai, or gemini.")

    # ------------------------------------------------------------------ #
    #  Message builder                                                     #
    # ------------------------------------------------------------------ #

    def _build_messages(self, query: str, context: str, history: list) -> list:
        """
        Build the message list to send to the LLM.
        Format:  system → history → new user message (with context injected).
        """
        user_message = (
            f"Context from the document:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}"
        )
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    # ------------------------------------------------------------------ #
    #  Provider implementations                                            #
    # ------------------------------------------------------------------ #

    def _call_groq(self, messages: list) -> str:
        from groq import Groq
        if self._client is None:
            self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = self._client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages,
            temperature=0.2,        # Low temperature = more factual, less creative
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _call_openai(self, messages: list) -> str:
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = self._client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _call_gemini(self, query: str, context: str) -> str:
        """Gemini has a different API style — no message history support here."""
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
