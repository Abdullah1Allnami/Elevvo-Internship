from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def preprocess_data(df):
    df = df.dropna().drop_duplicates()

    feature_cols = ["link", "headline", "short_description", "authors", "date"]
    X = df[feature_cols].astype(str).agg(" ".join, axis=1)
    X = clean_text(X)

    category_to_idx = {k: i for i, k in enumerate(df["category"].unique())}
    y = df["category"].map(category_to_idx).astype("float32")

    # Get both representations
    X_tfidf = get_embeddings(X, method="tfidf")
    X_seq = get_embeddings(X, method="sequence")

    # Split both
    X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )
    X_seq_train, X_seq_test, _, _ = train_test_split(
        X_seq, y, test_size=0.2, random_state=42
    )

    print("Data split into training and testing sets successfully.")
    return X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test


def clean_text(X):
    X = X.str.lower()
    X = X.str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
    X = X.str.replace(r"\@\w+|\#", "", regex=True)
    # Keep numbers and some punctuation
    X = X.str.replace(r"[^a-zA-Z0-9\s\.\?\!]", "", regex=True)
    X = X.str.replace(r"\s+", " ", regex=True)
    return X.str.strip()


def get_embeddings(X, method="tfidf", max_features=50000, max_len=500):
    if method == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words="english",
            min_df=5,
            max_df=0.7,
        )
        return vectorizer.fit_transform(X)

    elif method == "sequence":
        tokenizer = Tokenizer(
            num_words=max_features,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        return pad_sequences(
            sequences, maxlen=max_len, padding="post", truncating="post"
        )
