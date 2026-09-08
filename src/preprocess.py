from pathlib import Path
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATASET = DATA_DIR / "tickets.csv"
FALLBACK_DATASET = DATA_DIR / "all_tickets_processed_improved_v3.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def ensure_nltk_data() -> None:
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            else:
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_dataset() -> pd.DataFrame:
    source_path = RAW_DATASET if RAW_DATASET.exists() else FALLBACK_DATASET
    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected {RAW_DATASET} or {FALLBACK_DATASET}."
        )

    df = pd.read_csv(source_path)
    if not {"Document", "Topic_group"}.issubset(df.columns):
        expected_columns = ["Document", "Topic_group"]
        missing = [col for col in expected_columns if col not in df.columns]
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. Available columns: {list(df.columns)}"
        )
    return df[["Document", "Topic_group"]].copy()


def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set[str]) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = re.findall(r"[a-z]+", cleaned)
    tokens = [
        token for token in tokens if token not in stop_words and len(token) > 1]
    tokens = [lemmatizer.lemmatize(token, pos="n") for token in tokens]
    return " ".join(tokens)


def main() -> None:
    ensure_nltk_data()

    df = load_dataset()
    df["Document"] = df["Document"].fillna("")
    df["Topic_group"] = df["Topic_group"].fillna("Unknown")
    df = df.dropna(subset=["Document", "Topic_group"]).copy()
    df = df[df["Document"].astype(str).str.strip() != ""].copy()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    df["Document"] = df["Document"].apply(
        lambda text: clean_text(
            text, lemmatizer=lemmatizer, stop_words=stop_words)
    )
    df = df[df["Document"].astype(str).str.strip() != ""].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Topic_group"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved {len(train_df)} training rows to {TRAIN_PATH}")
    print(f"Saved {len(test_df)} testing rows to {TEST_PATH}")
    print(
        f"Dataset target distribution: {df['Topic_group'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
