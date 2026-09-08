from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "src" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "src" / "vectorizer.pkl"
CONFUSION_PATH = BASE_DIR / "app" / "confusion_matrix.png"

EXAMPLE_TICKETS = [
    "I cannot log in to my laptop because my password expired and I need it reset.",
    "My email is not syncing and I keep getting a verification warning after a password change.",
    "The monitor is flickering and the workstation will not power on after startup.",
    "I need access to the finance system approved by HR before the end of the week.",
    "The VPN connection keeps dropping while I am trying to connect to the internal server.",
]


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run src/train.py first.")

    model_bundle = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = model_bundle["model"] if isinstance(
        model_bundle, dict) and "model" in model_bundle else model_bundle
    accuracy = model_bundle.get("accuracy") if isinstance(
        model_bundle, dict) else None
    return model, vectorizer, accuracy


def predict_category(model, vectorizer, ticket_text: str):
    features = vectorizer.transform([ticket_text])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
    else:
        scores = model.decision_function(features)
        scores = np.asarray(scores)
        probabilities = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probabilities = probabilities / \
            probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities[0]

    class_index = int(np.argmax(probabilities))
    predicted_label = model.classes_[class_index]
    confidence = float(probabilities[class_index])
    return predicted_label, confidence, probabilities


st.set_page_config(page_title="IT Support Ticket Classifier", layout="wide")

st.title("IT Support Ticket Classifier")

try:
    model, vectorizer, accuracy = load_assets()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""

st.write("Paste a new IT support ticket below to route it to the most likely category.")

cols = st.columns(len(EXAMPLE_TICKETS))
for idx, ticket in enumerate(EXAMPLE_TICKETS):
    with cols[idx]:
        if st.button(f"Example {idx + 1}", key=f"example_{idx}"):
            st.session_state.ticket_input = ticket
            st.rerun()

user_input = st.text_area(
    "Ticket description",
    height=180,
    key="ticket_input",
)

if st.button("Predict category"):
    if user_input.strip():
        predicted_label, confidence, probabilities = predict_category(
            model, vectorizer, user_input)
        st.success(f"Predicted category: {predicted_label}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        probability_df = pd.DataFrame(
            {
                "Category": model.classes_,
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)
        st.dataframe(probability_df, hide_index=True, use_container_width=True)
    else:
        st.warning("Please enter a ticket description before predicting.")

st.sidebar.header("Model Summary")
if accuracy is not None:
    st.sidebar.metric("Overall test accuracy", f"{accuracy * 100:.2f}%")
else:
    st.sidebar.write("Accuracy is unavailable in the saved model bundle.")

if CONFUSION_PATH.exists():
    st.sidebar.image(str(CONFUSION_PATH), caption="Confusion matrix")
else:
    st.sidebar.write(
        "Confusion matrix image not found. Train the model to generate it.")
