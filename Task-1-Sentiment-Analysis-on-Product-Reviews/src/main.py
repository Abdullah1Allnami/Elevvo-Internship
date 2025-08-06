import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from src.data_downloader import data_downloader
from src.preprocess_data import preprocess_data
from src.train_models import train_models
from tests.evaluate import evaluate_model
import pandas as pd


def main():
    print("*" * 20, "Main", "*" * 20)
    data_downloader()
    print("Data downloaded successfully.")

    df = pd.read_csv("./data/IMDB Dataset.csv")
    print("Data loaded successfully.")

    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessed successfully.")

    models = train_models(X_train, y_train)
    print("Models trained successfully.")

    accuracy_results = {}
    for name, model in models.items():
        accuracy = evaluate_model(model, X_test, y_test)
        accuracy_results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}")

    print("Model evaluation completed.")
    print("Accuracy Results:", accuracy_results)
    print("*" * 20, "Return", "*" * 20)


if __name__ == "__main__":
    main()
