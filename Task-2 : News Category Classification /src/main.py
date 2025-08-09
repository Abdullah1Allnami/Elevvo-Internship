import json
import pandas as pd
from src.train_model import train_models
from tests.evaluate_model import evaluate_model
from src.preprocess_data import preprocess_data


def main():
    print("*" * 20, "Main", "*" * 20)
    print("\nStep 1: Loading dataset...")
    with open("./data/data.json", "r") as file:
        data = [json.loads(line) for line in file]

    # convert to pandas DataFrame
    data = pd.DataFrame(data)
    # select relevant columns
    data = data[["short_description", "category"]]

    print("Data loaded successfully.")

    # Step 2: Preprocess text
    print("\nStep 2: Preprocessing data...")
    X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test = (
        preprocess_data(data)
    )
    print("Data preprocessed successfully.")

    # Step 3: Train models
    print("\nStep 3: Training models...")
    models = train_models(X_tfidf_train, y_train, X_seq_train)
    print("Models trained successfully.")

    # Step 4: Evaluate models
    print("\nStep 4: Evaluating models...")
    accuracy_results = {}
    for name, model in models.items():
        if name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
            X_test = X_tfidf_test
        else:
            X_test = X_seq_test

        accuracy = evaluate_model(model, X_test, y_test)
        accuracy_results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")

    print("\nModel evaluation completed.")
    print("Accuracy Results:", accuracy_results)
    print("*" * 20, "Return", "*" * 20)


if __name__ == "__main__":
    main()
