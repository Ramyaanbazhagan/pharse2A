import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import tempfile
import speech_recognition as sr
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Neural Persona Call", layout="wide")

# ===============================
# GEMINI CONFIG
# ===============================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    genai.configure(api_key="YOUR_API_KEY_HERE")

# ===============================
# LOAD MEMORY
# ===============================
@st.cache_resource
def load_memory():
    with open("dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

memory_data = load_memory()

if "conversations" not in memory_data:
    memory_data["conversations"] = []

# ===============================
# FLATTEN MEMORY
# ===============================
def flatten_json(data):
    texts = []
    for k, v in data.items():
        if isinstance(v, list):
            for item in v:
                texts.append(str(item))
        else:
            texts.append(str(v))
    return texts

memory_texts = flatten_json(memory_data)

# ===============================
# EMBEDDINGS
# ===============================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(memory_texts, convert_to_tensor=True)
    return model, emb

model, embeddings = load_embeddings()

def retrieve_context(query, k=3):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, embeddings)[0]
    idx = np.argsort(-scores.cpu().numpy())[:k]
    return [memory_texts[i] for i in idx]

# ===============================
# EMOTION
# ===============================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry"]):
        return "comforting"
    if any(w in t for w in ["happy", "love", "great"]):
        return "happy"
    return "neutral"

# ===============================
# SPEAK
# ===============================
def speak(text):
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# ===============================
# LISTEN
# ===============================
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# ===============================
# UI
# ===============================
st.title("📞 Neural Persona – Call Mode")

if "call_active" not in st.session_state:
    st.session_state.call_active = False

if not st.session_state.call_active:
    if st.button("📞 Call"):
        st.session_state.call_active = True
        st.audio("ring.mp3")
        time.sleep(2)
        st.audio("ring.mp3")
        time.sleep(2)
        st.audio("ring.mp3")
        time.sleep(1)
        speak("Hello… I am here. You can talk to me.")

else:
    st.success("📞 Call Connected")

    if st.button("🎤 Speak"):
        user_text = listen()

        if user_text:
            st.write("🧍 You:", user_text)

            context = retrieve_context(user_text)
            context_text = "\n".join(context)

            prompt = f"""
You are a synthetic AI companion.
Speak naturally like a human in a phone call.
Never claim to be human.

Memory:
{context_text}

User: {user_text}
AI:
"""

            response = genai.GenerativeModel(
                "models/gemini-2.5-flash"
            ).generate_content(prompt)

            ai_text = response.text.strip()
            st.write("🤖 AI:", ai_text)
            speak(ai_text)

            memory_data["conversations"].append({
                "user": user_text,
                "ai": ai_text
            })

            with open("dataset.json", "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2)

    if st.button("❌ End Call"):
        st.session_state.call_active = False
        st.warning("Call Ended")
