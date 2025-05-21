# ── PATCH torch/classes + asyncio to prevent Streamlit crashes ──
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import nest_asyncio, asyncio, types, sys
nest_asyncio.apply()
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

try:
    import torch
    if not isinstance(torch.classes, types.ModuleType):
        dummy_mod = types.ModuleType("torch.classes")

        class _DummyPath(list):
            @property
            def _path(self): return []

        dummy_mod.__path__ = _DummyPath()
        torch.classes = dummy_mod
        sys.modules["torch.classes"] = dummy_mod
except Exception as e:
    print("[patch] torch.classes patch skipped:", e)
# ────────────────────────────────────────────────────────────────

import streamlit as st
from PIL import Image
from pathlib import Path
from propel_rag_chatbot import ask  # ← your RAG pipeline function

# Optional loader animation
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url: str):
    res = requests.get(url)
    if res.status_code != 200:
        return None
    return res.json()

lottie_loader = load_lottie_url("https://lottie.host/a9170e3d-4049-4617-acd9-ef2452f32619/2pbXyszbV1.json")


# App setup
st.set_page_config(page_title="🔧 Technical RAG Assistant", layout="wide")
st.title("🧠 Technical Manual Chatbot")

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input at bottom
user_input = st.chat_input("Type your question about the manual...")

if user_input:
    # Show user input in chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Loader animation while answering
    loader_placeholder = st.empty()
    with loader_placeholder:
        st_lottie(lottie_loader, height=40, width=40)

    # Run RAG pipeline
    try:
        answer, sources, image_path = ask(user_input)
    except Exception as e:
        answer = f"⚠️ Error: {e}"
        sources = []
        image_path = None

    loader_placeholder.empty()

    # Construct bot response
    full_response = answer
    if sources:
        full_response += "\n\n**🧾 Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    # Store and display bot message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.write(full_response)

        if image_path and Path(image_path).exists():
            st.image(Image.open(image_path), caption="📎 Relevant Diagram", use_container_width=False)
