# IMPORT LIBRARIES
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
print("Libraries imported successfully.")

# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# data and models folders
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(os.path.join(DATA_DIR, "final_processed_dataset.csv"))
print("Dataset loaded successfully.")

# =====================================================
# LOAD VECTORIZER
# =====================================================

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as file:
    vectorizer = pickle.load(file)
print("Vectorizer loaded successfully.")

# =====================================================
# LOAD INTENT MODEL
# =====================================================

with open(os.path.join(MODEL_DIR, "intent_model.pkl"), "rb") as file:
    intent_model = pickle.load(file)
print("Intent model loaded successfully.")

# =====================================================
# LOAD INTENT ENCODER
# =====================================================

with open(os.path.join(MODEL_DIR, "intent_encoder.pkl"), "rb") as file:
    intent_encoder = pickle.load(file)
print("Intent encoder loaded successfully.")

# =====================================================
# LOAD EMOTION MODEL
# =====================================================

with open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb") as file:
    emotion_model = pickle.load(file)
print("Emotion model loaded successfully.")

# =====================================================
# LOAD EMOTION ENCODER
# =====================================================

with open(os.path.join(MODEL_DIR, "emotion_encoder.pkl"), "rb") as file:
    emotion_encoder = pickle.load(file)
print("Emotion encoder loaded successfully.")

# PREPARE DATA
X = df["processed_text"]

# INTENT DATA
y_intent = intent_encoder.transform(
    df["intent"]
)

# EMOTION DATA

y_emotion = emotion_encoder.transform(
    df["emotion"]
)

# TRAIN TEST SPLIT
X_train,X_test,y_intent_train,y_intent_test = train_test_split(
    X,
    y_intent,
    test_size=0.2,
    random_state=42
)

# TF-IDF TRANSFORM

X_test_tfidf = vectorizer.transform(
    X_test
)

# INTENT PREDICTION
intent_predictions = intent_model.predict(
    X_test_tfidf
)

# INTENT ACCURACY

intent_accuracy = accuracy_score(
    y_intent_test,
    intent_predictions
)
print("\n==============================")
print("INTENT MODEL RESULTS")
print("==============================")
print(
    f"Accuracy: {intent_accuracy:.4f}"
)
#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
print("Total intent classes:",
      len(intent_encoder.classes_))

print("Classes in test set:",
      len(set(y_intent_test)))

print("Intent class names:")
print(intent_encoder.classes_)

# INTENT REPORT
intent_report = classification_report(
    y_intent_test,
    intent_predictions,
    labels=range(len(intent_encoder.classes_)),
    target_names=intent_encoder.classes_,
    zero_division=0
)
print(intent_report)

# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# metrics folder path
METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")

# create folder if it doesn't exist
os.makedirs(METRICS_DIR, exist_ok=True)

# save intent report
report_path = os.path.join(METRICS_DIR, "intent_report.txt")

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        f"Accuracy: {intent_accuracy:.4f}\n\n"
    )
    file.write(intent_report)

print("Intent report saved successfully!")

# INTENT CONFUSION MATRIX
intent_cm = confusion_matrix(
    y_intent_test,
    intent_predictions
)

plt.figure(figsize=(12,8))
sns.heatmap(
    intent_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title(
    "Intent Confusion Matrix"
)
plt.xlabel(
    "Predicted"
)
plt.ylabel(
    "Actual"
)
plt.tight_layout()
plt.show()
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# results folder path
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# create folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# save plot
plt.savefig(
    os.path.join(RESULTS_DIR, "intent_confusion_matrix.png")
)

print("Confusion matrix saved successfully!")
plt.close()

# EMOTION SPLIT
X_train,X_test,y_emotion_train,y_emotion_test = train_test_split(
    X,
    y_emotion,
    test_size=0.2,
    random_state=42
)

# TF-IDF TRANSFORM
X_test_tfidf = vectorizer.transform(
    X_test
)

# EMOTION PREDICTIONS
emotion_predictions = emotion_model.predict(
    X_test_tfidf
)

# EMOTION ACCURACY
emotion_accuracy = accuracy_score(
    y_emotion_test,
    emotion_predictions
)
print("\n==============================")
print("EMOTION MODEL RESULTS")
print("==============================")
print(
    f"Accuracy: {emotion_accuracy:.4f}"
)

# EMOTION REPORT
emotion_report = classification_report(
    y_emotion_test,
    emotion_predictions,
    labels=range(len(emotion_encoder.classes_)),
    target_names=emotion_encoder.classes_,
    zero_division=0
)
print(emotion_report)

# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# metrics folder path
METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")

# create folder if it doesn't exist
os.makedirs(METRICS_DIR, exist_ok=True)

# save emotion report
report_path = os.path.join(METRICS_DIR, "emotion_report.txt")

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        f"Accuracy: {emotion_accuracy:.4f}\n\n"
    )
    file.write(emotion_report)

print("Emotion report saved successfully!")
# EMOTION CONFUSION MATRIX
emotion_cm = confusion_matrix(
    y_emotion_test,
    emotion_predictions
)
plt.figure(figsize=(12,8))
sns.heatmap(
    emotion_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.title(
    "Emotion Confusion Matrix"
)
plt.xlabel(
    "Predicted"
)
plt.ylabel(
    "Actual"
)
plt.tight_layout()
plt.show()
plt.savefig(
    os.path.join(RESULTS_DIR, "emotional_confusion_matrix.png")
)

plt.close()

# HUMAN EVALUATION TEMPLATE
evaluation_data = {
    "User_Message": [
        "I feel stressed about exams",
        "Nobody understands me",
        "I feel anxious today",
        "I feel lonely",
        "I am happy today"
    ],
    "Relevance_Score": [

        "", "", "", "", ""
    ],
    "Supportive_Score": [

        "", "", "", "", ""
    ]
}
evaluation_df = pd.DataFrame(
    evaluation_data
)
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# results folder path
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# create folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# save CSV file
evaluation_df.to_csv(
    os.path.join(RESULTS_DIR, "human_evaluation_template.csv"),
    index=False
)

print("Human evaluation template saved successfully!")

# FINISHED
print("\nEvaluation Completed Successfully!")