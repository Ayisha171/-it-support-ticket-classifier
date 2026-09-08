# IT Support Ticket Classifier

Automatically classify IT support tickets into categories to speed up helpdesk routing using machine learning.

## Problem Statement

IT helpdesks receive hundreds of support tickets daily across multiple categories (Hardware, Access, HR Support, Storage, etc.). Manual ticket triage is time-consuming and error-prone. This project automates ticket classification using TF-IDF vectorization and a Linear SVM classifier, enabling faster routing and improved response times.

## Dataset

**Source:** [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) from Kaggle

**Contents:**

- 47,837 ticket records
- 8 target categories: Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, Administrative rights
- Features: Raw ticket text (Document column), category label (Topic_group column)

## Approach

This project uses a classical machine learning pipeline optimized for interpretability and speed:

1. **Text Preprocessing**
   - Lowercase conversion, punctuation/number removal
   - Stopword filtering and lemmatization
   - Stratified 80/20 train/test split

2. **Vectorization**
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Maximum 5,000 features for efficiency

3. **Model Selection**
   - Trained Logistic Regression and Linear SVM
   - Evaluated on macro-averaged metrics (F1, precision, recall)
   - Selected Linear SVM as the better performer

**Why classical ML over deep learning?**

- Fast inference and training
- Highly interpretable (feature importance)
- Excellent performance on short text classification
- Low computational overhead

## Results

**Test Set Performance (Linear SVM):**

- **Accuracy: 85.11%**
- **Macro F1: 85.58%**
- Macro Precision: 87.47%
- Macro Recall: 83.97%

Evaluated on 8 categories across 9,568 stratified test samples.

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Ayisha171/it-support-ticket-classifier.git
   cd it-support-ticket-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run Locally

### 1. Preprocess the Dataset

Generate cleaned train/test splits:

```bash
python src/preprocess.py
```

**Output:**

- `data/train.csv` (38,269 rows)
- `data/test.csv` (9,568 rows)

### 2. Train the Model

Train the TF-IDF vectorizer and Linear SVM classifier:

```bash
python src/train.py
```

**Output:**

- `src/model.pkl` — trained SVM model
- `src/vectorizer.pkl` — TF-IDF vectorizer
- `app/confusion_matrix.png` — confusion matrix visualization

### 3. Launch the Streamlit App

Start the interactive web app:

```bash
streamlit run app/app.py
```

Then open your browser to **http://localhost:8501**

## Usage

**In the Streamlit app:**

1. Click any "Example" button to auto-fill a sample ticket
2. Or paste your own ticket description in the text box
3. Click "Predict category" to see the classification
4. View confidence scores and per-category probabilities
5. Check the sidebar for model accuracy and confusion matrix

## Project Structure

```
it-support-ticket-classifier/
├── app/
│   └── app.py                    # Streamlit web interface
├── src/
│   ├── preprocess.py             # Data cleaning and splitting
│   ├── train.py                  # Model training and evaluation
│   ├── model.pkl                 # Trained SVM model (generated)
│   └── vectorizer.pkl            # TF-IDF vectorizer (generated)
├── data/
│   ├── all_tickets_processed_improved_v3.csv  # Original dataset
│   ├── tickets.csv               # Dataset file (symlink/copy)
│   ├── train.csv                 # Preprocessed training data (generated)
│   └── test.csv                  # Preprocessed test data (generated)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Dependencies

- **pandas** — data manipulation
- **numpy** — numerical computing
- **scikit-learn** — machine learning (TF-IDF, SVM, evaluation metrics)
- **nltk** — natural language processing (stopwords, lemmatization)
- **streamlit** — web interface framework
- **matplotlib** — visualization
- **seaborn** — statistical plotting
- **joblib** — model serialization

See `requirements.txt` for specific versions.

## Assumptions

- The dataset is available as `data/tickets.csv` or falls back to `data/all_tickets_processed_improved_v3.csv`
- Python 3.10+ environment with pip package manager
- Sufficient RAM for training (~2GB recommended)
- Labels are categorical and used as-is without additional encoding

## Model Files

The trained model and vectorizer are excluded from version control (see `.gitignore`) because they may exceed 50MB. To regenerate them:

```bash
python src/preprocess.py && python src/train.py
```

## License

This project is provided as-is for educational and commercial use.
