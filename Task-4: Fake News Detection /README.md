# Fake News Detection using Logistic Regression

This project implements a fake news detection system using logistic regression with comprehensive data preprocessing.

## Project Structure

```
Fake News Detection/
├── data/
│   ├── fake.csv          # Fake news dataset
│   └── true.csv          # True news dataset
├── src/
│   └── preprocess_data.py # Main preprocessing and modeling pipeline
├── test/
├── model/                # Directory for saved models
├── noteboooks/          # Jupyter notebooks (if any)
├── requirements.txt     # Python dependencies
├── test_preprocessing.py # Test script
└── README.md           # This file
```

## Features

- **Comprehensive Text Preprocessing**: 
  - URL and email removal
  - Special character cleaning
  - Text normalization
  - Duplicate removal

- **TF-IDF Vectorization**: 
  - Configurable n-gram ranges
  - Stop word removal
  - Feature selection based on frequency

- **Logistic Regression Model**:
  - Optimized hyperparameters
  - Cross-validation ready
  - Feature importance analysis

- **Model Evaluation**:
  - Accuracy metrics
  - Classification report
  - Confusion matrix
  - Feature importance visualization

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. The script will automatically download required NLTK data on first run.

## Usage

### Quick Start

Run the complete pipeline:
```bash
python src/preprocess_data.py
```

### Test the Pipeline

Run the test script to verify everything works:
```bash
python test_preprocessing.py
```

### Using the Preprocessor Class

```python
from src.preprocess_data import FakeNewsPreprocessor

# Initialize preprocessor
preprocessor = FakeNewsPreprocessor()

# Load and preprocess data
df = preprocessor.load_data('data/fake.csv', 'data/true.csv')
df_processed = preprocessor.preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)

# Vectorize text
X_train_tfidf, X_test_tfidf = preprocessor.vectorize_text(X_train, X_test)

# Train model
model = preprocessor.train_logistic_regression(X_train_tfidf, y_train)

# Evaluate
y_pred, accuracy = preprocessor.evaluate_model(model, X_test_tfidf, y_test)
```

## Data Format

The CSV files should contain the following columns:
- `title`: News article title
- `text`: News article content
- `subject`: News category (optional)
- `date`: Publication date (optional)

## Model Performance

The logistic regression model typically achieves:
- High accuracy on the test set
- Good precision and recall for both fake and true news
- Interpretable feature importance

## Customization

You can customize the preprocessing by modifying the `FakeNewsPreprocessor` class:

- **TF-IDF Parameters**: Adjust `max_features`, `ngram_range`, `min_df`, `max_df`
- **Text Cleaning**: Modify the `clean_text()` method
- **Model Parameters**: Change logistic regression hyperparameters in `train_logistic_regression()`

## Next Steps

1. **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters
2. **Feature Engineering**: Add more text features (sentiment, readability scores)
3. **Model Comparison**: Try other algorithms (Random Forest, SVM, Neural Networks)
4. **Model Deployment**: Create a web API for real-time predictions

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- nltk >= 3.6.0
