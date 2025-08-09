import pandas as pd
import warnings
from src.data_downloader import data_downloader
from src.preprocess_data import preprocess_data
from src.train_models import train_models
from tests.evaluate import evaluate_model
from src.visualize_most_fre_words import visualize_most_frequent_words

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("*" * 20, "Main", "*" * 20)

    # Step 1: Download data
    print("\nStep 1: Loading dataset...")
    df = pd.read_csv("/data/IMDB Dataset.csv")
    print("Data loaded successfully.")

    # Step 2: Preprocess text
    print("\nStep 2: Preprocessing data...")
    X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test = (
        preprocess_data(df)
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

    # Step 5: Visualize most frequent words
    print("\nStep 5: Visualizing most frequent words...")
    visualize_most_frequent_words(df)
    print("Visualization completed.")

    print("*" * 20, "Return", "*" * 20)


if __name__ == "__main__":
    main()
