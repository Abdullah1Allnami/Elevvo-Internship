from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    # Example cleaning function, modify as needed
    X = [x.str.replace(r"[^a-zA-Z\s]", "", regex=True) for x in X]
    print("Text data cleaned successfully.")
    return X


def get_embeddings(X, method="tfidf", max_features=20000, max_len=500):
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_vec = vectorizer.fit_transform(X)
        print("TF-IDF embeddings generated successfully.")
        return X_vec

    elif method == "sequence":
        tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        padded = pad_sequences(
            sequences, maxlen=max_len, padding="post", truncating="post"
        )
        print("Sequence embeddings generated successfully.")
        return padded
