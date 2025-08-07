from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import numpy as np


def preprocess_data(df):
    df = df.dropna()
    X = clean_text(df["review"])
    y = (
        df.loc[X.index, "sentiment"]
        .map({"positive": 1, "negative": 0})
        .astype("float32")
    )

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
    X = X.str.replace(r"[^a-zA-Z\s]", "", regex=True)
    lemmaizer = WordNetLemmatizer()
    X = X.apply(lambda x: " ".join([lemmaizer.lemmatize(word) for word in x.split()]))
    X = X.str.lower()
    X = X.str.strip()
    X = X.str.replace(r"\s+", " ", regex=True)

    # remove stop word
    stop_words = set(
        [
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        ]
    )
    X = X.apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )

    # remove extra spaces
    X = X.str.replace(r"\s+", " ", regex=True).str.strip()

    # remove empty strings
    X = X[X != ""]

    # remove duplicates
    X = X.drop_duplicates()

    # reset index
    X = X.reset_index(drop=True)

    print("Text data cleaned successfully.")
    return X


def get_embeddings(X, method="tfidf", max_features=30000, max_len=700):
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

    # Uncomment the following lines if you want to load pre-trained embeddings
    # Note: This requires additional libraries and files, so it's commented out by default.
    elif method == "pretrained":
        glove_embeddings = {}
        with open("path/to/glove.6B.300d.txt", "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                glove_embeddings[word] = coefs
        print("Pre-trained embeddings loaded successfully.")
        return glove_embeddings

    else:
        raise ValueError("Invalid method specified. Use 'tfidf' or 'sequence'.")
