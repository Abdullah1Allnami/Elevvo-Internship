#!/usr/bin/env python3
"""
Training script for fake news detection models
"""

import sys
import os
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from get_model import get_model, get_model_info
from preprocess_data import FakeNewsPreprocessor

class ModelTrainer:
    def __init__(self, model_type="logistic", save_model=True, model_dir="../model"):
        """
        Initialize the model trainer
        
        Args:
            model_type (str): Type of model to train
            save_model (bool): Whether to save the trained model
            model_dir (str): Directory to save models
        """
        self.model_type = model_type
        self.save_model = save_model
        self.model_dir = model_dir
        self.preprocessor = FakeNewsPreprocessor()
        self.model = None
        self.training_history = {}
        
        # Create model directory if it doesn't exist
        if save_model and not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def load_and_preprocess_data(self, fake_path, true_path):
        """Load and preprocess the data"""
        print(f"Loading and preprocessing data for {self.model_type} model...")
        
        # Load data
        df = self.preprocessor.load_data(fake_path, true_path)
        
        # Preprocess data
        df_processed = self.preprocessor.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(df_processed)
        
        # For traditional ML models, use TF-IDF
        if self.model_type in ["logistic", "svm", "random_forest", "naive_bayes", "xgboost"]:
            X_train_processed, X_test_processed = self.preprocessor.vectorize_text(X_train, X_test)
            return X_train_processed, X_test_processed, y_train, y_test, None
        
        # For neural networks, use tokenization
        else:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Tokenize text
            tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
            tokenizer.fit_on_texts(X_train)
            
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            
            # Pad sequences
            max_length = 200
            X_train_processed = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
            X_test_processed = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
            
            return X_train_processed, X_test_processed, y_train, y_test, tokenizer
    
    def create_model(self, X_train, tokenizer=None):
        """Create the model based on type"""
        print(f"Creating {self.model_type} model...")
        
        if self.model_type in ["logistic", "svm", "random_forest", "naive_bayes", "xgboost"]:
            self.model = get_model(self.model_type)
        else:
            # Neural network models
            max_features = 10000 if tokenizer is None else len(tokenizer.word_index) + 1
            max_length = 200
            input_dim = max_features  # Add input_dim parameter
            self.model = get_model(self.model_type, input_dim=input_dim, max_features=max_features, max_length=max_length)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        
        if self.model_type in ["logistic", "svm", "random_forest", "naive_bayes", "xgboost"]:
            # Traditional ML models
            self.model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
        else:
            # Neural network models
            if X_val is None or y_val is None:
                # Split training data for validation
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            else:
                X_train_split, X_val_split, y_train_split, y_val_split = X_train, X_val, y_train, y_val
            
            # Train the model
            history = self.model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val_split, y_val_split),
                epochs=10,
                batch_size=32,
                verbose=1
            )
            
            self.training_history = history.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print(f"Evaluating {self.model_type} model...")
        
        # Make predictions
        if self.model_type in ["logistic", "svm", "random_forest", "naive_bayes", "xgboost"]:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        else:
            y_pred_proba = self.model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba = y_pred_proba[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                print(f"ROC AUC Score: {roc_auc:.4f}")
            except:
                print("Could not calculate ROC AUC score")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """Plot training history for neural networks"""
        if not self.training_history:
            print("No training history available for plotting")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['accuracy'], label='Training Accuracy')
        plt.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_metadata(self, tokenizer=None, evaluation_results=None):
        """Save the trained model and metadata"""
        if not self.save_model:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_type}_model_{timestamp}"
        
        # Save model
        if self.model_type in ["logistic", "svm", "random_forest", "naive_bayes", "xgboost"]:
            # Save sklearn model
            model_path = os.path.join(self.model_dir, f"{model_filename}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            # Save Keras model
            model_path = os.path.join(self.model_dir, f"{model_filename}.h5")
            self.model.save(model_path)
        
        # Save tokenizer if available
        if tokenizer is not None:
            tokenizer_path = os.path.join(self.model_dir, f"{model_filename}_tokenizer.pkl")
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'timestamp': timestamp,
            'model_info': get_model_info(self.model_type),
            'training_history': self.training_history,
            'evaluation_results': evaluation_results
        }
        
        metadata_path = os.path.join(self.model_dir, f"{model_filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for traditional ML models"""
        if self.model_type not in ["logistic", "svm", "random_forest", "naive_bayes", "xgboost"]:
            print("Hyperparameter tuning not implemented for neural network models")
            return self.model
        
        print(f"Performing hyperparameter tuning for {self.model_type}...")
        
        # Define parameter grids
        param_grids = {
            "logistic": {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            },
            "svm": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf']
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            "naive_bayes": {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            "xgboost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if self.model_type not in param_grids:
            print(f"No hyperparameter grid defined for {self.model_type}")
            return self.model
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model,
            param_grids[self.model_type],
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model

def main():
    """Main function to train models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--model', type=str, default='logistic', 
                       choices=['logistic', 'svm', 'random_forest', 'naive_bayes', 'xgboost', 
                               'lstm', 'bilstm', 'cnn', 'transformer'],
                       help='Model type to train')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    
    args = parser.parse_args()
    
    # File paths
    fake_path = "/Users/hassanali/Desktop/project/Fake News Detection/data/fake.csv"
    true_path = "/Users/hassanali/Desktop/project/Fake News Detection/data/true.csv"
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_type=args.model,
        save_model=not args.no_save
    )
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tokenizer = trainer.load_and_preprocess_data(fake_path, true_path)
    
    # Create model
    trainer.create_model(X_train, tokenizer)
    
    # Hyperparameter tuning (optional)
    if args.tune:
        trainer.hyperparameter_tuning(X_train, y_train)
    
    # Train model
    trainer.train_model(X_train, y_train)
    
    # Evaluate model
    evaluation_results = trainer.evaluate_model(X_test, y_test)
    
    # Plot training history (for neural networks)
    if trainer.training_history:
        trainer.plot_training_history()
    
    # Save model and metadata
    trainer.save_model_and_metadata(tokenizer, evaluation_results)
    
    print(f"\n✅ Training completed for {args.model} model!")
    print(f"Final accuracy: {evaluation_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
