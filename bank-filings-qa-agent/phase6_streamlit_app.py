"""
Phase 6, Step 3: Streamlit frontend.

Concept: this file knows NOTHING about embeddings, Chroma, LangGraph,
or Groq. Its only job is: take user input, send it to the FastAPI
/ask endpoint, and display whatever comes back. This is the
backend/frontend separation in action - swap this file entirely
for a different UI and the backend wouldn't need to change at all.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Bank Filings Q&A", page_icon="🏦")
st.title("🏦 Bank Filings Q&A Agent")
st.caption("Ask questions about JPMorgan, Bank of America, and Wells Fargo's real 10-K filings.")

# st.session_state persists data across reruns within the same
# browser session - Streamlit reruns the whole script top-to-bottom
# on every interaction, so without this, chat history would vanish
# every time you asked a new question.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the full conversation history on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# st.chat_input renders a chat-style text box pinned to the bottom
question = st.chat_input("Ask about loan loss provisions, net income, risk factors...")

if question:
    # Show the user's question immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Call the FastAPI backend - this is the ONLY place this file
    # talks to the rest of your system, through a plain HTTP request.
    with st.chat_message("assistant"):
        with st.spinner("Searching filings and verifying evidence..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                needs_review = data["needs_review"]
                sources = data["sources"]

                st.write(answer)

                if needs_review:
                    st.warning("⚠️ Low confidence - this answer was flagged for human review.")

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- {s['ticker']}, filed {s['filing_date']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.ConnectionError:
                error_msg = "Could not reach the API. Is `uvicorn phase6_api:app` running?"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})