import pandas as pd
import warnings
from src.data_downloader import data_downloader
from src.preprocess_data import preprocess_data
from src.train_models import train_models
from tests.evaluate import evaluate_model

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("*" * 20, "Main", "*" * 20)

    # Step 1: Download data
    data_downloader()
    print("Data downloaded successfully.")

    # Step 2: Load and limit dataset (for debugging or testing)
    df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
    # df = df.head(10)
    print("Data loaded successfully.")

    # Step 3: Preprocess text → returns TF-IDF and Sequence embeddings
    X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test = (
        preprocess_data(df)
    )
    print("Data preprocessed successfully.")

    # Step 4: Train all models
    models = train_models(X_tfidf_train, y_train, X_seq_train)
    print("Models trained successfully.")

    # Step 5: Evaluate each model using appropriate test input
    accuracy_results = {}
    for name, model in models.items():
        if name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
            X_test = X_tfidf_test
        else:
            X_test = X_seq_test

        accuracy = evaluate_model(model, X_test, y_test)
        accuracy_results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}")

    print("Model evaluation completed.")
    print("Accuracy Results:", accuracy_results)
    print("*" * 20, "Return", "*" * 20)


if __name__ == "__main__":
    main()
