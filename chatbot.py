print('Phase 6 --> Transformer Chatbot')
#import libraries
import os
import pickle
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import warnings
warnings.filterwarnings('ignore')
print('Libraries imported successfully!')
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)
# models folder path
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# load TF-IDF vectorizer
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as file:
    vectorizer = pickle.load(file)
    print('TF-IDF vectorizer loaded successfully!')
# load intent model
with open(os.path.join(MODEL_DIR, "intent_model.pkl"), "rb") as file:
    intent_model = pickle.load(file)
    print('Intent model loaded successfully!')
# load intent encoder
with open(os.path.join(MODEL_DIR, "intent_encoder.pkl"), "rb") as file:
    intent_encoder = pickle.load(file)
    print('Intent label encoder loaded successfully!')
# load emotion model
with open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb") as file:
    emotion_model = pickle.load(file)
    print('Emotion model loaded successfully!')
# load emotion encoder
with open(os.path.join(MODEL_DIR, "emotion_encoder.pkl"), "rb") as file:
    emotion_encoder = pickle.load(file)
    print('Emotion label encoder loaded successfully!')
#conversation memory
chat_history_ids = None
#load microsoft DialoGPT model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
print('DialoGPT model and tokenizer loaded successfully!')
#predict intent function
def predict_intent(text):
    vector = vectorizer.transform([text])
    prediction = intent_model.predict(vector)
    intent = intent_encoder.inverse_transform(
        prediction
    )[0]
    return intent
#predict emotion function
def predict_emotion(text):
    vector = vectorizer.transform([text])
    prediction = emotion_model.predict(vector)
    emotion = emotion_encoder.inverse_transform(
        prediction
    )[0]
    return emotion
#generate dialogue response function
def generate_dialogpt_response(text,emotion):
    global chat_history_ids
    prompt = f"""
User emotion: {emotion}
User says:
{text}
Respond supportively:
"""
    new_input_ids = tokenizer.encode(
        prompt + tokenizer.eos_token,
        return_tensors="pt"
    )
    bot_input_ids = (
        torch.cat(
            [chat_history_ids, new_input_ids],
            dim=-1
        )
        if chat_history_ids is not None
        else new_input_ids
    )
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.6
    )
    response = tokenizer.decode(
        chat_history_ids[
            :,
            bot_input_ids.shape[-1]:
        ][0],
        skip_special_tokens=True
    )
    return response
#create supportive response
emotion_templates = {
    "sadness":
    "I'm sorry you're feeling sad. ",
    "anxiety":
    "It's understandable to feel anxious. ",
    "stress":
    "Stress can feel overwhelming sometimes. ",
    "loneliness":
    "Feeling lonely can be difficult. ",
    "depression":
    "I'm here to listen and support you. ",
    "neutral":
    ""
}
#main chatbot function
def chatbot_response(user_text):
    intent = predict_intent(user_text)
    emotion = predict_emotion(user_text)
    generated_response = generate_dialogpt_response(user_text,emotion)
    supportive_prefix = emotion_templates.get(emotion,"")
    final_response = (
        supportive_prefix
        +
        generated_response
    )
    return {
        "intent": intent,
        "emotion": emotion,
        "response": final_response
    }
#test chatbot with sample input
user_input = "I feel stressed because of exams"
result = chatbot_response(user_input)
print("Intent:")
print(result["intent"])
print("\nEmotion:")
print(result["emotion"])
print("\nResponse:")
print(result["response"])
#interactive chat loop
print("""
==================================================

AI Mental Health Support Chatbot

Type 'exit' to quit.

This chatbot provides emotional support only.
It is not a licensed therapist.

==================================================
""")

while True:
    user_input = input('\n'"You: ")
    if user_input.lower() == "exit":
        print("\nChat ended.")
        break
    result = chatbot_response(
        user_input
    )
    print("\nIntent :", result["intent"])
    print("Emotion:", result["emotion"])
    print("\nChatbot:")
    print(result["response"])