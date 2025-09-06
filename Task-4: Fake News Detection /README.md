# Fake News Detection with Logistic Regression

A robust pipeline for detecting fake news using logistic regression, featuring advanced text preprocessing and model evaluation. This repository includes a comprehensive workflow notebook for step-by-step exploration.

## Repository Structure

```
Fake News Detection/
├── data/
│   ├── fake.csv          # Fake news dataset
│   └── true.csv          # True news dataset
├── src/
│   └── preprocess_data.py # Preprocessing and modeling pipeline
├── model/                # Saved models
├── notebooks/
│   └── workflow.ipynb    # End-to-end workflow notebook
├── requirements.txt      # Python dependencies
├── test/
│   └── test_preprocessing.py # Test script
└── README.md             # Project documentation
```

## Key Features

- **Advanced Text Preprocessing**
  - Removal of URLs, emails, and special characters
  - Text normalization and duplicate elimination

- **TF-IDF Vectorization**
  - Configurable n-gram ranges and stop word removal
  - Feature selection based on frequency

- **Logistic Regression Modeling**
  - Optimized hyperparameters
  - Cross-validation support
  - Feature importance analysis

- **Comprehensive Evaluation**
  - Accuracy, classification report, confusion matrix
  - Feature importance visualization

- **Interactive Workflow Notebook**
  - Explore the entire pipeline in `notebooks/workflow.ipynb`

## Getting Started

### Installation

Install dependencies:
```bash
pip install -r requirements.txt
```
NLTK data will be downloaded automatically on first run.

### Usage

#### Run the Pipeline

```bash
python src/preprocess_data.py
```

#### Run Tests

```bash
python test/test_preprocessing.py
```

#### Explore the Workflow

Open and run the notebook for a guided walkthrough:
```
notebooks/workflow.ipynb
```

## Data Format

CSV files must include:
- `title`: Article title
- `text`: Article content
- `subject`: News category (optional)
- `date`: Publication date (optional)

## Model Performance

- High test accuracy
- Strong precision and recall for both fake and true news
- Interpretable feature importance

## Customization

- **TF-IDF**: Adjust `max_features`, `ngram_range`, `min_df`, `max_df`
- **Text Cleaning**: Edit `clean_text()` in `FakeNewsPreprocessor`
- **Model Parameters**: Modify logistic regression settings

## Next Steps

- Hyperparameter tuning (GridSearchCV)
- Feature engineering (sentiment, readability)
- Model comparison (Random Forest, SVM, Neural Networks)
- Deployment (web API for real-time predictions)

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- nltk >= 3.6.0
