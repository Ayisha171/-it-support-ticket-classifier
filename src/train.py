from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
APP_DIR = BASE_DIR / "app"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = BASE_DIR / "src" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "src" / "vectorizer.pkl"
CONFUSION_PATH = APP_DIR / "confusion_matrix.png"


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Train/test split not found. Run src/preprocess.py first.")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))

    report = {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions,
        "cm": cm,
        "classes": np.unique(y_test),
    }

    print(f"\n{name} Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))
    print("Confusion Matrix:")
    print(cm)
    return report


def plot_confusion_matrix(cm: np.ndarray, labels: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    train_df, test_df = load_split_data()

    X_train = train_df["Document"].astype(str)
    X_test = test_df["Document"].astype(str)
    y_train = train_df["Topic_group"]
    y_test = test_df["Topic_group"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    logistic_model = LogisticRegression(max_iter=2000, random_state=42)
    logistic_model.fit(X_train_tfidf, y_train)

    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    logistic_report = evaluate_model("Logistic Regression", logistic_model, X_test_tfidf, y_test)
    svm_report = evaluate_model("Linear SVM", svm_model, X_test_tfidf, y_test)

    if logistic_report["f1"] >= svm_report["f1"]:
        best_name = "Logistic Regression"
        best_model = logistic_model
        best_report = logistic_report
    else:
        best_name = "Linear SVM"
        best_model = svm_model
        best_report = svm_report

    plot_confusion_matrix(best_report["cm"], best_report["classes"], f"{best_name} Confusion Matrix")

    model_payload = {
        "model": best_model,
        "accuracy": float(best_report["accuracy"]),
        "f1": float(best_report["f1"]),
        "label_order": list(best_model.classes_),
    }
    joblib.dump(model_payload, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nSelected model: {best_name}")
    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved confusion matrix plot to {CONFUSION_PATH}")
    print(f"Best model accuracy: {best_report['accuracy']:.4f}")
    print(f"Best model macro F1: {best_report['f1']:.4f}")


if __name__ == "__main__":
    main()
