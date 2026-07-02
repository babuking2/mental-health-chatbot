# =====================================================
# PHASE 8: GRADIO UI FOR AI MENTAL HEALTH CHATBOT
# =====================================================

import gradio as gr
import os
import pickle
import torch

# =====================================================
# LOAD MODELS
# =====================================================
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)
# models folder path
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
# load intent model
intent_model = pickle.load(
    open(os.path.join(MODEL_DIR, "intent_model.pkl"), "rb")
)
# load emotion model
emotion_model = pickle.load(
    open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb")
)
# load TF-IDF vectorizer
vectorizer = pickle.load(
    open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb")
)
# load intent encoder
intent_encoder = pickle.load(
    open(os.path.join(MODEL_DIR, "intent_encoder.pkl"), "rb")
)
# load emotion encoder
emotion_encoder = pickle.load(
    open(os.path.join(MODEL_DIR, "emotion_encoder.pkl"), "rb")
)

# =====================================================
# LOAD DIALO GPT
# =====================================================
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

chat_history_ids = None

# =====================================================
# SUPPORT TEMPLATES
# =====================================================

emotion_templates = {
    "sad": "I'm really sorry you're feeling sad. ",
    "stress": "Stress can feel overwhelming. ",
    "anxiety": "It's okay to feel anxious sometimes. ",
    "loneliness": "Feeling lonely can be hard. ",
    "neutral": ""
}

# =====================================================
# CRISIS DETECTION
# =====================================================

def detect_crisis(text):
    text = text.lower()
    keywords = ["suicide", "kill myself", "self harm", "die"]
    return any(k in text for k in keywords)

# =====================================================
# MAIN CHAT FUNCTION
# =====================================================

def chatbot_response(user_input, history):

    global chat_history_ids

    # ---------------- CRISIS CHECK ----------------
    if detect_crisis(user_input):
        return (
            "⚠ I am really concerned about you. "
            "Please reach out to a trusted person or professional immediately."
        )

    # ---------------- INTENT ----------------
    intent_vec = vectorizer.transform([user_input])
    intent_pred = intent_model.predict(intent_vec)
    intent = intent_encoder.inverse_transform(intent_pred)[0]

    # ---------------- EMOTION ----------------
    emotion_pred = emotion_model.predict(intent_vec)
    emotion = emotion_encoder.inverse_transform(emotion_pred)[0]

    # ---------------- PROMPT ----------------
    prompt = f"""
You are a supportive mental health chatbot.

User emotion: {emotion}

User message: {user_input}

Give a kind supportive response:
"""

    new_input_ids = tokenizer.encode(
        prompt + tokenizer.eos_token,
        return_tensors="pt"
    )

    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.6
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    prefix = emotion_templates.get(emotion, "")

    final_response = prefix + response

    return f"""
Intent: {intent}
Emotion: {emotion}

Bot: {final_response}
"""

# =====================================================
# GRADIO UI
# =====================================================

with gr.Blocks() as app:

    gr.Markdown("""
    # 🧠 AI Mental Health Support Chatbot

    ⚠ **Disclaimer:**  
    This chatbot provides emotional support only.  
    It is NOT a licensed therapist or medical professional.
    """)

    chatbot = gr.ChatInterface(
        fn=chatbot_response
    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    app.launch()