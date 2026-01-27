import streamlit as st
import json, time
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Neural Persona", layout="wide")

# -------------------------------
# API CONFIG
# -------------------------------
genai.configure(api_key="YOUR_GEMINI_KEY")

# -------------------------------
# INIT STATES
# -------------------------------
if "call_state" not in st.session_state:
    st.session_state.call_state = "ringing"

if "chat" not in st.session_state:
    st.session_state.chat = []

if "call_start" not in st.session_state:
    st.session_state.call_start = None

# -------------------------------
# LOAD MEMORY
# -------------------------------
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# -------------------------------
# FLATTEN MEMORY
# -------------------------------
def flatten_json(data):
    out = []
    for k, v in data.items():
        if isinstance(v, list):
            out.extend([str(i) for i in v])
        else:
            out.append(str(v))
    return out

memory_texts = flatten_json(memory_data)

# -------------------------------
# EMBEDDINGS
# -------------------------------
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(memory_texts, convert_to_tensor=True)
    return model, emb

model, embeddings = load_embeddings()

def retrieve_context(query):
    q = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q, embeddings)[0]
    top = np.argsort(-scores.cpu().numpy())[:3]
    return [memory_texts[i] for i in top]

# -------------------------------
# EMOTION
# -------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["miss", "sad", "lonely"]):
        return "😢 Sad"
    if any(w in t for w in ["happy", "love"]):
        return "😊 Happy"
    return "😐 Neutral"

# -------------------------------
# VOICE (Fallback-safe)
# -------------------------------
def speak(text):
    tts = gTTS(text, lang="en")
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(f.name)
    return f.name

# -------------------------------
# UI
# -------------------------------
st.title("📞 Neural Persona – AI Call Simulation")

# ===============================
# CALL STATES
# ===============================
if st.session_state.call_state == "ringing":
    st.subheader("Incoming Call")
    st.audio("ring.mp3", autoplay=True)

    col1, col2 = st.columns(2)
    if col1.button("✅ Accept"):
        st.session_state.call_state = "connected"
        st.session_state.call_start = time.time()
        st.rerun()

    if col2.button("❌ Reject"):
        st.session_state.call_state = "ended"
        st.rerun()

elif st.session_state.call_state == "connected":
    st.subheader("📞 Call Connected")

    hello = "Hello… I’m here. Talk to me."
    st.audio(speak(hello), autoplay=True)

    if st.button("🎤 Speak"):
        st.session_state.call_state = "talking"
        st.rerun()

elif st.session_state.call_state == "talking":
    elapsed = int(time.time() - st.session_state.call_start)
    st.info(f"⏱ Call Duration: {elapsed} sec")

    user_text = st.text_input("You:")

    if st.button("Send Voice/Text"):
        ctx = retrieve_context(user_text)
        prompt = f"""
You are a comforting AI persona.
Context: {ctx}
User: {user_text}
AI:
"""
        response = genai.GenerativeModel(
            "models/gemini-2.5-flash"
        ).generate_content(prompt)

        ai_text = response.text
        emotion = detect_emotion(ai_text)

        st.session_state.chat.append(("You", user_text))
        st.session_state.chat.append(("AI", ai_text))

        st.markdown(f"**AI ({emotion}):** {ai_text}")
        st.audio(speak(ai_text), autoplay=True)

    if st.button("❌ End Call"):
        st.session_state.call_state = "ended"
        st.rerun()

elif st.session_state.call_state == "ended":
    st.subheader("📴 Call Ended")
    st.write("Conversation Saved.")
