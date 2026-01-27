import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Neural Persona",
    layout="wide"
)

# -------------------------------
# GEMINI API CONFIG (CORRECT)
# -------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# -------------------------------
# LOAD MEMORY
# -------------------------------
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

# -------------------------------
# FLATTEN JSON
# -------------------------------
def flatten_json(data, parent_key=""):
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.append((f"{new_key}[{i}]", str(item)))
        else:
            items.append((new_key, str(v)))
    return items

memory_pairs = flatten_json(memory_data)
memory_texts = [v for _, v in memory_pairs]

# -------------------------------
# EMBEDDINGS
# -------------------------------
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(memory_texts, convert_to_tensor=True)
    return model, embeddings

model, embeddings = load_embeddings()

# -------------------------------
# MEMORY RETRIEVAL
# -------------------------------
def retrieve_context(query, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_idx = np.argsort(-scores.cpu().numpy())[:top_k]
    return [memory_texts[i] for i in top_idx]

# -------------------------------
# SIMPLE EMOTION DETECTION
# -------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "upset", "cry"]):
        return "comforting"
    if any(w in t for w in ["happy", "great", "love", "excited"]):
        return "happy"
    return "neutral"

# -------------------------------
# TEXT TO SPEECH
# -------------------------------
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🧠 Control Panel")
voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)
show_reasoning = st.sidebar.checkbox("🧠 Show Reasoning")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Synthetic AI Persona • Research Prototype"
)

# -------------------------------
# MAIN UI
# -------------------------------
st.title("🤖 Neural Persona – Face-to-Face AI")

col1, col2 = st.columns([1, 2])

with col1:
    st.image(
        "https://i.imgur.com/8Km9tLL.png",
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

# -------------------------------
# INPUT
# -------------------------------
st.markdown("---")
user_input = st.text_input("Talk to your companion:")

if st.button("Send"):
    if user_input.strip():
        st.session_state.chat.append(("user", user_input))

        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        prompt = f"""
You are a synthetic AI persona created for academic research.
You speak warmly and emotionally but never claim to be human.

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

        st.session_state.chat.append(("ai", ai_text))

        if voice_on:
            audio_file = speak(ai_text)
            st.audio(audio_file)

        if show_reasoning:
            st.markdown("### 🧠 Reasoning Trace")
            st.write("**Emotion detected:**", emotion)
            st.write("**Memory used:**")
            for c in context:
                st.write("-", c)
