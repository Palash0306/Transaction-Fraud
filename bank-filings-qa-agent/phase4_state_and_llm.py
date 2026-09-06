"""
Phase 4, Step 1: Shared state + LLM connection.

Concept: LangGraph passes a single "state" object through every
node. Each node reads whatever it needs from state and returns a
dict of NEW/UPDATED fields - LangGraph merges those into the state
automatically before handing it to the next node.

We define state as a TypedDict: basically a dictionary with a fixed,
known set of keys and their types, so your editor (and you) can see
exactly what data flows through the graph at every step.
"""

import os
import re
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found - check your .env file.")


class GraphState(TypedDict):
    """
    The single shared object that flows through every node.

    question:        the user's original question (set at the start)
    retrieved_chunks: list of chunks from retrieve_context() (Node 1 fills this)
    verification:     dict with the verifier's judgment (Node 2 fills this)
    answer:           the final synthesized answer (Node 3 fills this)
    needs_review:     True if confidence was too low to answer confidently
    """
    question: str
    retrieved_chunks: List[dict]
    verification: dict
    answer: str
    needs_review: bool


# Two LLM connections - a smaller/faster model for the verifier
# (a narrower yes/no-ish judgment call), a stronger model for the
# final answer (where quality matters most to the end user).
verifier_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=api_key,
    temperature=0,
    max_tokens=1024,
)

synthesis_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0,
    max_tokens=1024,
)


def strip_thinking(raw_response: str) -> str:
    """
    Removes <think>...</think> reasoning blocks some models produce,
    leaving just the final answer text.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    return cleaned.strip()


def extract_json_object(text: str) -> dict | None:
    """
    Finds and parses the first valid JSON block containing 'confidence'.
    """
    # Find the outermost JSON block containing 'confidence'
    match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback to standard curly brace search
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None