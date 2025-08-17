import json
import pandas as pd
from src.train_model import train_and_evaluate_models
from src.preprocess_data import preprocess_data


def main():
    print("*" * 20, "Main", "*" * 20)
    print("\nStep 1: Loading dataset...")
    with open(
        "/kaggle/input/news-category-dataset/News_Category_Dataset_v3.json", "r"
    ) as file:
        data = [json.loads(line) for line in file]

    # convert to pandas DataFrame
    data = pd.DataFrame(data)

    # data = data.head(100) # For testing

    print("Data loaded successfully.")

    # Step 2: Preprocess text
    print("\nStep 2: Preprocessing data...")
    X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test = (
        preprocess_data(data)
    )
    print("Data preprocessed successfully.")

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
    )
    print("Models Trained and Evaluated successfully.")
    print("*" * 20, "Return", "*" * 20)


if __name__ == "__main__":
    main()
