import json
import pandas as pd
from src.train_model import train_and_evaluate_models
from src.train_model import train_and_evaluate_models
from src.preprocess_data import preprocess_data
from sklearn.model_selection import train_test_split


def main():
    print("*" * 20, "Main", "*" * 20)
    print("\nStep 1: Loading dataset...")
    with open(
        "/kaggle/input/news-category-dataset/News_Category_Dataset_v3.json", "r"
    ) as file:
    with open(
        "/kaggle/input/news-category-dataset/News_Category_Dataset_v3.json", "r"
    ) as file:
        data = [json.loads(line) for line in file]

    # convert to pandas DataFrame
    data = pd.DataFrame(data)
    feature_cols = ["link", "headline", "short_description", "authors", "date"]

    category_to_idx = {k: i for i, k in enumerate(data["category"].unique())}
    y = data["category"].map(category_to_idx).astype("float32")

    print("Data loaded successfully.")

    # Step 2: Preprocess text
    print("\nStep 2: Preprocessing data...")
    X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test = (
        preprocess_data(data, feature_cols, target_col="category")
    )

    print("Data preprocessed successfully.")

    original_text = data[feature_cols].astype(str).agg(" ".join, axis=1)
    y = data["category"].astype("float32")
    original_text_train, original_text_test, _, _ = train_test_split(
        original_text, y, test_size=0.2, random_state=42
    )

    # Step 3: Train models
    print("\nStep 3: Training and Evaluating models...")
    models = train_and_evaluate_models(
        X_tfidf_train,
        X_tfidf_test,
        X_seq_train,
        X_seq_test,
        y_train,
        y_test,
        num_classes=data["category"].nunique(),
        original_text_train=original_text_train,
        original_text_test=original_text_test,
    )
    print("Models Trained and Evaluated successfully.")
    print("*" * 20, "Return", "*" * 20)


if __name__ == "__main__":
    main()
