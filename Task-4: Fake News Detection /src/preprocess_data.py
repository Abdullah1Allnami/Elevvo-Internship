import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FakeNewsPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
    def load_data(self, fake_path, true_path):
        """Load fake and true news datasets"""
        print("Loading datasets...")
        
        # Load fake news data
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 0  # 0 for fake news
        
        # Load true news data  
        true_df = pd.read_csv(true_path)
        true_df['label'] = 1  # 1 for true news
        
        # Combine datasets
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        print(f"Fake news samples: {len(fake_df)}")
        print(f"True news samples: {len(true_df)}")
        print(f"Total samples: {len(combined_df)}")
        
        return combined_df
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_data(self, df):
        """Main preprocessing function"""
        print("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        print("Handling missing values...")
        df_processed = df_processed.dropna(subset=['title', 'text'])
        
        # Remove duplicates
        print("Removing duplicates...")
        df_processed = df_processed.drop_duplicates()
        
        # Clean text columns
        print("Cleaning text data...")
        df_processed['title_clean'] = df_processed['title'].apply(self.clean_text)
        df_processed['text_clean'] = df_processed['text'].apply(self.clean_text)
        
        # Combine title and text for better feature representation
        df_processed['combined_text'] = df_processed['title_clean'] + ' ' + df_processed['text_clean']
        
        # Remove rows with empty combined text
        df_processed = df_processed[df_processed['combined_text'].str.len() > 10]
        
        print(f"Final dataset shape: {df_processed.shape}")
        print(f"Label distribution:\n{df_processed['label'].value_counts()}")
        
        return df_processed
    
    def vectorize_text(self, X_train, X_test):
        """Convert text to TF-IDF vectors"""
        print("Vectorizing text data...")
        
        # Fit TF-IDF on training data
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        print(f"TF-IDF matrix shape - Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("Splitting data into train/test sets...")
        
        X = df['combined_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train_tfidf, y_train):
        """Train logistic regression model"""
        print("Training Logistic Regression model...")
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        
        model.fit(X_train_tfidf, y_train)
        
        print("Model training completed!")
        return model
    
    def evaluate_model(self, model, X_test_tfidf, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'True']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return y_pred, accuracy
    
    def plot_results(self, y_test, y_pred):
        """Plot evaluation results"""
        # Confusion Matrix Heatmap
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Feature importance (top 20)
        plt.subplot(1, 2, 2)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        feature_importance = np.abs(self.model.coef_[0])
        top_indices = np.argsort(feature_importance)[-20:]
        
        plt.barh(range(20), feature_importance[top_indices])
        plt.yticks(range(20), [feature_names[i] for i in top_indices])
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the complete pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fake News Detection - Data Preprocessing and Logistic Regression')
    parser.add_argument('--model', type=str, default='logistic', 
                       choices=['logistic', 'svm', 'random_forest', 'naive_bayes'],
                       help='Model type to train (default: logistic)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')
    parser.add_argument('--save-data', action='store_true', help='Save processed data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FAKE NEWS DETECTION - DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
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
        
        # Save processed data if requested
        if args.save_data:
            output_path = "/Users/hassanali/Desktop/project/Fake News Detection/data/processed_data.csv"
            df_processed.to_csv(output_path, index=False)
            print(f"Processed data saved to: {output_path}")
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        # Vectorize text
        X_train_tfidf, X_test_tfidf = preprocessor.vectorize_text(X_train, X_test)
        
        # Train model based on argument
        if args.model == 'logistic':
            model = preprocessor.train_logistic_regression(X_train_tfidf, y_train)
        else:
            # For other models, use the get_model function
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
            from get_model import get_model
            
            model = get_model(args.model)
            print(f"Training {args.model} model...")
            model.fit(X_train_tfidf, y_train)
            print("Model training completed!")
        
        preprocessor.model = model  # Store model for plotting
        
        # Evaluate model
        y_pred, accuracy = preprocessor.evaluate_model(model, X_test_tfidf, y_test)
        
        # Plot results if not disabled
        if not args.no_plot:
            preprocessor.plot_results(y_test, y_pred)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Model: {args.model}")
        print(f"Final Accuracy: {accuracy:.4f}")
        print("=" * 60)
        
        return model, preprocessor
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, preprocessor = main()
