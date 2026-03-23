# chat_chain.py
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---- Configuration (via env, with safe defaults) ----
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
MODEL_ID: str = os.environ.get("MODEL_ID", "openai/gpt-oss-20b:nebius")
BASE_URL: str = os.environ.get("BASE_URL", "https://router.huggingface.co/v1")
TEMP: float = float(os.environ.get("TEMPERATURE", "0.2"))


# ---- Build the chain once (module-level cache) ----
# If HF_TOKEN is not provided, fall back to a simple local stub so the
# backend can run without external credentials in development.
_llm = None
_chain = None

if HF_TOKEN:
    _llm = ChatOpenAI(
        model=MODEL_ID,
        api_key=HF_TOKEN,
        base_url=BASE_URL,
        temperature=TEMP,
    )

    _prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, precise assistant. Reply in simple, neutral English."),
        ("user", "{message}")
    ])

    _chain = _prompt | _llm | StrOutputParser()


def get_answer(message: str) -> str:
    """
    Generate a single reply for the given user message.
    Keeps LangChain initialization separate from the web layer.
    If HF_TOKEN is not configured, returns a safe fallback string.
    """
    if not message or not message.strip():
        raise ValueError("message cannot be empty.")

    if _chain is None:
        # fallback behavior when HF_TOKEN is not set
        return "[assistant disabled] Set HF_TOKEN to enable AI replies."

    return _chain.invoke({"message": message.strip()})
