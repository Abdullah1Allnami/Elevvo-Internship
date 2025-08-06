# Sentiment Analysis on Product Reviews

A comprehensive pipeline for sentiment analysis on IMDB movie reviews using machine learning and deep learning models.

## Overview

This project classifies movie reviews as positive or negative. It covers:
- Data acquisition and cleaning
- Feature extraction (TF-IDF and sequence embeddings)
- Model training (Logistic Regression, Random Forest, Naive Bayes, LSTM, GRU, Transformer-like)
- Model evaluation

## Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Fields:** `review`, `sentiment`

## Structure

```
Task-1-Sentiment-Analysis-on-Product-Reviews/
├── data/                # Data files
├── notebooks/           # Jupyter notebooks (EDA, workflow)
├── src/                 # Source code
│   ├── data_downloader.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   └── main.py
├── tests/               # Evaluation scripts
├── README.md
```

## Installation

```bash
git clone [<repo-url>](https://github.com/Abdullah1Allnami/Elevvo-Internship.git)
cd Task-1-Sentiment-Analysis-on-Product-Reviews
pip install -r requirements.txt
```

## Usage

1. Run the main pipeline:
   ```bash
   python -m src/main.py
   ```
   This will download,  preprocess data, train models, and print evaluation results.

## Notebooks

- **EDA.ipynb:** Data exploration and visualization.
- **task-1-sentiment-analysis-on-product-reviews.ipynb:** Full workflow and experiments.

## Contributing

Pull requests and issues are welcome.

## License

MIT License.
