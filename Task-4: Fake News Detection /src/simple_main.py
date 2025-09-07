#!/usr/bin/env python3
"""
Simplified Fake News Detection System - TensorFlow Free Version

This version focuses on traditional ML models that are more reliable
and don't have the TensorFlow compatibility issues.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings

warnings.filterwarnings("ignore")


class SimpleFakeNewsDetector:
    def __init__(self, data_dir="data"):
        """Initialize the simple fake news detector"""
        self.data_dir = data_dir
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.model = None
        self.is_trained = False

    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def load_data(self):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")

        # Load fake news data
        fake_path = os.path.join(self.data_dir, "fake.csv")
        true_path = os.path.join(self.data_dir, "true.csv")

        fake_df = pd.read_csv(fake_path)
        fake_df["label"] = 0  # 0 for fake news

        # Load true news data
        true_df = pd.read_csv(true_path)
        true_df["label"] = 1  # 1 for true news

        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)

        print(f"Dataset loaded: {len(df)} total samples")
        print(f"   Fake news: {len(fake_df)} samples")
        print(f"   True news: {len(true_df)} samples")

        # Handle missing values
        df = df.dropna(subset=["title", "text"])

        # Remove duplicates
        df = df.drop_duplicates()

        # Clean text columns
        print("Cleaning text data...")
        df["title_clean"] = df["title"].apply(self.clean_text)
        df["text_clean"] = df["text"].apply(self.clean_text)

        # Combine title and text
        df["combined_text"] = df["title_clean"] + " " + df["text_clean"]

        # Remove rows with empty combined text
        df = df[df["combined_text"].str.len() > 10]

        print(f"Data preprocessing completed!")
        print(f"   Final dataset: {len(df)} samples")
        print(f"   Label distribution: {df['label'].value_counts().to_dict()}")

        return df

    def train_model(self, model_type="logistic"):
        """Train the specified model"""
        print(f"Training {model_type} model...")

        # Load and preprocess data
        df = self.load_data()

        # Prepare features and labels
        X = df["combined_text"]
        y = df["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorize text
        print("Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print(f"   Training set: {X_train_tfidf.shape}")
        print(f"   Test set: {X_test_tfidf.shape}")

        # Create and train model
        if model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000, random_state=42, C=1.0, solver="liblinear"
            )
        elif model_type == "svm":
            self.model = SVC(kernel="linear", C=1.0, random_state=42, probability=True)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
            )
        elif model_type == "naive_bayes":
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        print("Training model...")
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate model
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model training completed!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Fake", "True"]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"   True Negatives (Fake→Fake): {cm[0,0]}")
        print(f"   False Positives (Fake→True): {cm[0,1]}")
        print(f"   False Negatives (True→Fake): {cm[1,0]}")
        print(f"   True Positives (True→True): {cm[1,1]}")

        self.is_trained = True
        return accuracy

    def predict(self, text):
        """Predict if a news article is fake or true"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")

        # Clean the text
        cleaned_text = self.clean_text(text)

        # Vectorize
        text_vectorized = self.vectorizer.transform([cleaned_text])

        # Predict
        prediction = self.model.predict(text_vectorized)[0]

        # Get probability if available
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(text_vectorized)[0]
            confidence = max(probability)
        else:
            confidence = None

        result = {
            "prediction": "Fake" if prediction == 0 else "True",
            "confidence": confidence,
            "fake_probability": probability[0] if confidence else None,
            "true_probability": probability[1] if confidence else None,
        }

        return result

    def interactive_mode(self):
        """Interactive prediction mode"""
        print("Interactive Prediction Mode")
        print("=" * 50)
        print("Enter news articles to classify (type 'quit' to exit)")
        print()

        while True:
            text = input("Enter news article: ").strip()

            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not text:
                print("Please enter some text.")
                continue

            try:
                result = self.predict(text)
                print(f"\nPrediction: {result['prediction']}")
                if result["confidence"]:
                    print(f"Confidence: {result['confidence']:.4f}")
                    print(f"Fake Probability: {result['fake_probability']:.4f}")
                    print(f"True Probability: {result['true_probability']:.4f}")
                print("-" * 50)
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Simple Fake News Detection System (TensorFlow Free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_main.py --train logistic              # Train logistic regression
  python simple_main.py --train random_forest        # Train random forest
  python simple_main.py --predict "Your news text"   # Predict single article
  python simple_main.py --interactive                # Interactive mode
        """,
    )

    parser.add_argument(
        "--train",
        type=str,
        choices=["logistic", "svm", "random_forest", "naive_bayes"],
        help="Train a specific model",
    )
    parser.add_argument(
        "--predict", type=str, help="Predict if a news article is fake or true"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive prediction mode"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Data directory path"
    )

    args = parser.parse_args()

    print("Simple Fake News Detection System")
    print("=" * 50)
    print("No TensorFlow dependencies - more reliable!")
    print()

    # Initialize detector
    detector = SimpleFakeNewsDetector(data_dir=args.data_dir)

    try:
        if args.train:
            # Train model
            accuracy = detector.train_model(model_type=args.train)
            print(f"\nTraining completed with {accuracy:.4f} accuracy!")

        elif args.predict:
            # Train a default model first
            if not detector.is_trained:
                print("Training default model first...")
                detector.train_model(model_type="logistic")

            # Make prediction
            result = detector.predict(args.predict)
            print(f"\nPrediction: {result['prediction']}")
            if result["confidence"]:
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Fake Probability: {result['fake_probability']:.4f}")
                print(f"True Probability: {result['true_probability']:.4f}")

        elif args.interactive:
            # Train a default model first
            if not detector.is_trained:
                print("🤖 Training default model first...")
                detector.train_model(model_type="logistic")

            # Start interactive mode
            detector.interactive_mode()

        else:
            # No action specified, show help
            parser.print_help()
            print("\n💡 Quick start:")
            print("   python simple_main.py --train logistic")
            print("   python simple_main.py --interactive")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
