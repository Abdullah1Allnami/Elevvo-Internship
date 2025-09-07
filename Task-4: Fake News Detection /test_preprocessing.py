#!/usr/bin/env python3
"""
Test script for the fake news detection preprocessing pipeline
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess_data import FakeNewsPreprocessor


def test_preprocessing():
    """Test the preprocessing pipeline with a small sample"""
    print("Testing Fake News Detection Preprocessing Pipeline")
    print("=" * 50)

    # Initialize preprocessor
    preprocessor = FakeNewsPreprocessor()

    # File paths
    fake_path = "/Users/hassanali/Desktop/project/Fake News Detection/data/fake.csv"
    true_path = "/Users/hassanali/Desktop/project/Fake News Detection/data/true.csv"

    try:
        # Load data
        df = preprocessor.load_data(fake_path, true_path)

        # Preprocess data
        df_processed = preprocessor.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)

        # Vectorize text
        X_train_tfidf, X_test_tfidf = preprocessor.vectorize_text(X_train, X_test)

        # Train model
        model = preprocessor.train_logistic_regression(X_train_tfidf, y_train)

        # Evaluate model
        y_pred, accuracy = preprocessor.evaluate_model(model, X_test_tfidf, y_test)

        print(f"\nPipeline completed successfully!")
        print(f"Final accuracy: {accuracy:.4f}")

        return True

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_preprocessing()
    if success:
        print("\nAll tests passed! Your preprocessing pipeline is ready.")
    else:
        print("\nTests failed. Please check the error messages above.")
