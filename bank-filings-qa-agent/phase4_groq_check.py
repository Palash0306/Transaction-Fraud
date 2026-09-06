"""
Sanity check: confirm your Groq API key actually works.
 
Concept: python-dotenv reads your .env file and loads its contents
as environment variables, so os.environ.get("GROQ_API_KEY") can
find it - this is how you keep secrets out of your actual code.
 
This script does ONE simple thing: send a basic question to Groq's
LLM and print the response. If this works, we know the key, the
network, and the library are all fine before adding any complexity.
"""

import os
import requests
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")


# def get_active_groq_models(api_key: str):
#     """Fetches valid models directly from the Groq API endpoint."""
#     url = "https://api.groq.com/openai/v1/models"
#     headers = {"Authorization": f"Bearer {api_key}"}

#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         print(f"Failed to fetch models: {response.status_code} - {response.text}")
#         return []

#     models_data = response.json().get("data", [])
#     model_ids = [m["id"] for m in models_data]
#     return model_ids


# 1. Fetch live models
# active_models = get_active_groq_models(api_key)
# print("=== ACTIVE GROQ MODELS FOR YOUR KEY ===")
# for model in active_models:
#     print(f" - {model}")

# # 2. Automatically test the first available model
# if active_models:
#     target_model = active_models[0]
#     print(f"\nTesting ChatGroq with active model: '{target_model}'...")

#     try:
#         llm = ChatGroq(model=target_model, api_key=API_KEY, temperature=0)
#         res = llm.invoke("Reply with 'SUCCESS'")
#         print(f"Response: {res.content}")
#     except Exception as e:
#         print(f"Inference test failed: {e}")
# else:
#     print("\nNo active models returned. Please check your GROQ_API_KEY.")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check that your .env file exists "
        "in the project root and contains a line like: "
        "GROQ_API_KEY=your_actual_key_here"
    )
print("API key loaded (first 6 chars):", api_key[:6] + "...")
 
# llama-3.1-8b-instant is a small, fast, free-tier-friendly model on
# Groq - good for this quick check. You can swap models later.
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=api_key,
    temperature=0,
    max_tokens=1024,
)

def strip_thinking(raw_response:str)-> str:
    """
    Removes the model's <think>...</think> reasoning block, leaving
    only the final answer. Reasoning models like this one expose
    their internal thought process - useful for debugging, but not
    something we want to show end users as "the answer."
    """
    cleaned = re.sub(r"<think>.*?</think>","",raw_response,flags=re.DOTALL)
    return cleaned.strip()

 
response = llm.invoke("In one sentence, what is a 10-K filing?")
 
print("\n--- RAW response (includes thinking) ---")
print(response.content)
 
print("\n--- CLEANED response (thinking stripped) ---")
print(strip_thinking(response.content))



