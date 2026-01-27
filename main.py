import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Neural Persona",
    layout="wide"
)

# ===============================
# GEMINI API CONFIG
# ===============================
# (Works with hard-coded key OR Streamlit Secrets)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    genai.configure(api_key="AIzaSyC5gHEcGQMxZIom4Tz-0aXv6b4mQOX1SGc")

# ===============================
# LOAD MEMORY
# ===============================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# Ensure conversations exist
if "conversations" not in memory_data:
    memory_data["conversations"] = []

# ===============================
# FLATTEN JSON
# ===============================
def flatten_json(data, parent_key=""):
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key))
        elif isinstance(v, list):
            for item in v:
                items.append((new_key, str(item)))
        else:
            items.append((new_key, str(v)))
    return items

memory_pairs = flatten_json(memory_data)
memory_texts = [v for _, v in memory_pairs]

# ===============================
# EMBEDDINGS
# ===============================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

model, embeddings = load_embeddings()

# ===============================
# MEMORY RETRIEVAL
# ===============================
def retrieve_context(query, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# ===============================
# EMOTION DETECTION
# ===============================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "upset"]):
        return "comforting"
    if any(w in t for w in ["happy", "love", "great", "excited"]):
        return "happy"
    return "neutral"

# ===============================
# SAVE LONG-TERM MEMORY
# ===============================
def save_memory(user, ai):
    memory_data["conversations"].append({
        "user": user,
        "ai": ai
    })
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

# ===============================
# TEXT TO SPEECH
# ===============================
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🧠 Control Panel")
voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Reasoning")

st.sidebar.markdown("---")
st.sidebar.caption("Synthetic AI Persona • Academic Research")

# ===============================
# MAIN UI
# ===============================
st.title("🤖 Neural Persona – Face-to-Face AI")

col1, col2 = st.columns([1, 2])

# Avatar logic
avatar_map = {
    "happy": "https://i.imgur.com/H5oZ6vT.png",
    "comforting": "https://i.imgur.com/4M7IWwP.png",
    "neutral": "https://i.imgur.com/8Km9tLL.png"
}

if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

with col1:
    st.image(
        avatar_map[st.session_state.current_emotion],
        caption="AI Persona",
        use_column_width=True
    )

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"🧍 **You:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")

# ===============================
# INPUT
# ===============================
st.markdown("---")
user_input = st.text_input("Talk to your companion:")

if st.button("Send"):
    if user_input.strip():
        st.session_state.chat.append(("user", user_input))

        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        prompt = f"""
You are a synthetic AI persona created for academic research.
Never claim to be human or a real person.
Your emotional tone should be: {st.session_state.current_emotion}

Memory Context:
{context_text}

User: {user_input}
AI:
"""

        response = genai.GenerativeModel(
            "models/gemini-2.5-flash"
        ).generate_content(prompt)

        ai_text = response.text.strip()

        emotion = detect_emotion(ai_text)
        st.session_state.current_emotion = emotion

        st.session_state.chat.append(("ai", ai_text))

        save_memory(user_input, ai_text)

        if voice_on:
            audio = speak(ai_text)
            st.audio(audio)

        if show_reasoning:
            st.markdown("### 🧠 Reasoning Trace")
            st.write("Emotion:", emotion)
            st.write("Memory used:")
            for c in context:
                st.write("-", c)
