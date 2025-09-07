#!/usr/bin/env python3
"""
Main entry point for Fake News Detection System

This script provides a unified interface for:
- Data preprocessing
- Model training
- Model evaluation
- Predictions
- Model comparison
"""

import sys
import os
import argparse
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

from preprocess_data import FakeNewsPreprocessor
from train_model import ModelTrainer
from get_model import get_model, get_model_info, compare_models


class FakeNewsDetectionSystem:
    def __init__(self, data_dir="../data", model_dir="../model"):
        """
        Initialize the Fake News Detection System

        Args:
            data_dir (str): Directory containing the data files
            model_dir (str): Directory to save/load models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.preprocessor = None
        self.trained_models = {}

        # File paths
        self.fake_path = os.path.join(data_dir, "fake.csv")
        self.true_path = os.path.join(data_dir, "true.csv")

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    def preprocess_data(self, save_processed=False):
        """
        Preprocess the fake news data

        Args:
            save_processed (bool): Whether to save processed data

        Returns:
            tuple: (X_train, X_test, y_train, y_test, df_processed)
        """
        print("🔄 Starting data preprocessing...")
        print("=" * 50)

        # Initialize preprocessor
        self.preprocessor = FakeNewsPreprocessor()

        # Load data
        df = self.preprocessor.load_data(self.fake_path, self.true_path)

        # Preprocess data
        df_processed = self.preprocessor.preprocess_data(df)

        # Save processed data if requested
        if save_processed:
            processed_path = os.path.join(self.data_dir, "processed_data.csv")
            df_processed.to_csv(processed_path, index=False)
            print(f"✅ Processed data saved to: {processed_path}")

        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(df_processed)

        # Vectorize text
        X_train_tfidf, X_test_tfidf = self.preprocessor.vectorize_text(X_train, X_test)

        print(f"✅ Data preprocessing completed!")
        print(f"   Training set: {X_train_tfidf.shape}")
        print(f"   Test set: {X_test_tfidf.shape}")

        return X_train_tfidf, X_test_tfidf, y_train, y_test, df_processed

    def train_model(
        self, model_type="logistic", tune_hyperparams=False, save_model=True
    ):
        """
        Train a specific model

        Args:
            model_type (str): Type of model to train
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
            save_model (bool): Whether to save the trained model

        Returns:
            dict: Training results
        """
        print(f"🤖 Training {model_type} model...")
        print("=" * 50)

        # Preprocess data if not already done
        if self.preprocessor is None:
            X_train, X_test, y_train, y_test, _ = self.preprocess_data()
        else:
            # Use existing preprocessor
            df = self.preprocessor.load_data(self.fake_path, self.true_path)
            df_processed = self.preprocessor.preprocess_data(df)
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(
                df_processed
            )
            X_train, X_test = self.preprocessor.vectorize_text(X_train, X_test)

        # Initialize trainer
        trainer = ModelTrainer(
            model_type=model_type, save_model=save_model, model_dir=self.model_dir
        )

        # Create model
        trainer.create_model(X_train)

        # Hyperparameter tuning if requested
        if tune_hyperparams:
            trainer.hyperparameter_tuning(X_train, y_train)

        # Train model
        trainer.train_model(X_train, y_train)

        # Evaluate model
        evaluation_results = trainer.evaluate_model(X_test, y_test)

        # Save model and metadata
        if save_model:
            model_path, metadata_path = trainer.save_model_and_metadata(
                evaluation_results=evaluation_results
            )
            print(f"✅ Model saved to: {model_path}")

        # Store results
        self.trained_models[model_type] = {
            "model": trainer.model,
            "trainer": trainer,
            "results": evaluation_results,
        }

        print(f"✅ {model_type} model training completed!")
        print(f"   Accuracy: {evaluation_results['accuracy']:.4f}")

        return evaluation_results

    def compare_models(self, model_types=None, tune_hyperparams=False):
        """
        Train and compare multiple models

        Args:
            model_types (list): List of model types to compare
            tune_hyperparams (bool): Whether to perform hyperparameter tuning

        Returns:
            dict: Comparison results
        """
        if model_types is None:
            model_types = ["logistic", "svm", "random_forest", "naive_bayes"]

        print("🔍 Comparing multiple models...")
        print("=" * 50)

        results = {}

        for model_type in model_types:
            print(f"\n📊 Training {model_type}...")
            try:
                result = self.train_model(
                    model_type=model_type,
                    tune_hyperparams=tune_hyperparams,
                    save_model=False,  # Don't save individual models during comparison
                )
                results[model_type] = result
            except Exception as e:
                print(f"❌ Error training {model_type}: {str(e)}")
                results[model_type] = {"accuracy": 0.0, "error": str(e)}

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1].get("accuracy", 0))
        print(
            f"\n🏆 Best Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}"
        )

        return results

    def predict(self, text, model_type=None):
        """
        Predict if a news article is fake or true

        Args:
            text (str): News article text
            model_type (str): Model to use for prediction (uses best if None)

        Returns:
            dict: Prediction results
        """
        if self.preprocessor is None:
            print("❌ No preprocessor available. Please run preprocessing first.")
            return None

        if not self.trained_models:
            print("❌ No trained models available. Please train a model first.")
            return None

        # Use specified model or best available model
        if model_type is None:
            model_type = max(
                self.trained_models.keys(),
                key=lambda x: self.trained_models[x]["results"]["accuracy"],
            )

        if model_type not in self.trained_models:
            print(
                f"❌ Model {model_type} not found. Available models: {list(self.trained_models.keys())}"
            )
            return None

        model = self.trained_models[model_type]["model"]

        # Clean the text
        cleaned_text = self.preprocessor.clean_text(text)

        # Vectorize
        text_vectorized = self.preprocessor.tfidf_vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vectorized)[0]

        # Get probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(text_vectorized)[0]
            confidence = max(probability)
        else:
            confidence = None

        result = {
            "prediction": "Fake" if prediction == 0 else "True",
            "confidence": confidence,
            "fake_probability": probability[0] if confidence else None,
            "true_probability": probability[1] if confidence else None,
            "model_used": model_type,
        }

        return result

    def interactive_prediction(self):
        """
        Interactive prediction mode
        """
        print("🔮 Interactive Prediction Mode")
        print("=" * 50)
        print("Enter news articles to classify (type 'quit' to exit)")
        print()

        while True:
            text = input("📰 Enter news article: ").strip()

            if text.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not text:
                print("❌ Please enter some text.")
                continue

            result = self.predict(text)
            if result:
                print(f"\n🎯 Prediction: {result['prediction']}")
                if result["confidence"]:
                    print(f"📊 Confidence: {result['confidence']:.4f}")
                    print(f"� Fake Probability: {result['fake_probability']:.4f}")
                    print(f"📈 True Probability: {result['true_probability']:.4f}")
                print(f"🤖 Model Used: {result['model_used']}")
                print("-" * 50)

    def show_model_info(self, model_type=None):
        """
        Show information about available models

        Args:
            model_type (str): Specific model type to show info for
        """
        if model_type:
            info = get_model_info(model_type)
            print(f"📋 Model Information: {info['name']}")
            print("=" * 50)
            print(f"Type: {info['type']}")
            print(f"Description: {info['description']}")
            print(f"Pros: {', '.join(info['pros'])}")
            print(f"Cons: {', '.join(info['cons'])}")
        else:
            compare_models()

    def save_session(self, filename=None):
        """
        Save current session (models and preprocessor)

        Args:
            filename (str): Filename to save to
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

        session_data = {
            "timestamp": datetime.now().isoformat(),
            "trained_models": list(self.trained_models.keys()),
            "model_dir": self.model_dir,
            "data_dir": self.data_dir,
        }

        session_path = os.path.join(self.model_dir, filename)
        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)

        print(f"✅ Session saved to: {session_path}")

    def load_session(self, filename):
        """
        Load a previous session

        Args:
            filename (str): Filename to load from
        """
        session_path = os.path.join(self.model_dir, filename)

        if not os.path.exists(session_path):
            print(f"❌ Session file not found: {session_path}")
            return

        with open(session_path, "r") as f:
            session_data = json.load(f)

        print(f"✅ Session loaded from: {session_path}")
        print(f"   Timestamp: {session_data['timestamp']}")
        print(f"   Trained models: {session_data['trained_models']}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Fake News Detection System - Main Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --preprocess                    # Preprocess data only
  python main.py --train logistic                # Train logistic regression
  python main.py --compare                       # Compare multiple models
  python main.py --predict "Your news text here" # Predict single article
  python main.py --interactive                   # Interactive prediction mode
  python main.py --info                          # Show model information
        """,
    )

    # Main actions
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument(
        "--train",
        type=str,
        choices=[
            "logistic",
            "svm",
            "random_forest",
            "naive_bayes",
            "xgboost",
            "lstm",
            "bilstm",
            "cnn",
            "transformer",
        ],
        help="Train a specific model",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple models"
    )
    parser.add_argument(
        "--predict", type=str, help="Predict if a news article is fake or true"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive prediction mode"
    )

    # Options
    parser.add_argument(
        "--tune", action="store_true", help="Perform hyperparameter tuning"
    )
    parser.add_argument("--save-data", action="store_true", help="Save processed data")
    parser.add_argument("--model", type=str, help="Specify model type for prediction")
    parser.add_argument(
        "--info",
        type=str,
        nargs="?",
        const="all",
        help='Show model information (specify model type or use "all")',
    )

    # File paths
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Data directory path"
    )
    parser.add_argument(
        "--model-dir", type=str, default="../model", help="Model directory path"
    )

    args = parser.parse_args()

    # Initialize system
    system = FakeNewsDetectionSystem(data_dir=args.data_dir, model_dir=args.model_dir)

    print("🚀 Fake News Detection System")
    print("=" * 50)

    try:
        # Handle different actions
        if args.preprocess:
            system.preprocess_data(save_processed=args.save_data)

        elif args.train:
            system.train_model(model_type=args.train, tune_hyperparams=args.tune)

        elif args.compare:
            system.compare_models(tune_hyperparams=args.tune)

        elif args.predict:
            # Preprocess and train a default model if needed
            if system.preprocessor is None:
                print("Preprocessing data...")
                system.preprocess_data()

            if not system.trained_models:
                print("Training default model...")
                system.train_model(model_type="logistic")

            result = system.predict(args.predict, model_type=args.model)
            if result:
                print(f"\nPrediction: {result['prediction']}")
                if result["confidence"]:
                    print(f"Confidence: {result['confidence']:.4f}")
                print(f"Model Used: {result['model_used']}")

        elif args.interactive:
            # Preprocess and train a default model if needed
            if system.preprocessor is None:
                print("Preprocessing data...")
                system.preprocess_data()

            if not system.trained_models:
                print("Training default model...")
                system.train_model(model_type="logistic")

            system.interactive_prediction()

        elif args.info:
            if args.info == "all":
                system.show_model_info()
            else:
                system.show_model_info(args.info)

        else:
            # No action specified, show help
            parser.print_help()
            print("\n Quick start:")
            print("   python main.py --preprocess --train logistic")
            print("   python main.py --compare")
            print("   python main.py --interactive")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
