# Fake News Detection System - Usage Guide

## Quick Start

The `main.py` file provides a unified interface for all fake news detection functionality.

### Basic Usage Examples

#### 1. Preprocess Data Only
```bash
python3 src/main.py --preprocess
```

#### 2. Train a Single Model
```bash
# Train Logistic Regression
python3 src/main.py --train logistic

# Train Random Forest with hyperparameter tuning
python3 src/main.py --train random_forest --tune

# Train XGBoost
python3 src/main.py --train xgboost
```

#### 3. Compare Multiple Models
```bash
# Compare traditional ML models
python3 src/main.py --compare

# Compare with hyperparameter tuning
python3 src/main.py --compare --tune
```

#### 4. Make Predictions
```bash
# Predict a single article
python3 src/main.py --predict "This is a sample news article to classify."

# Use a specific model for prediction
python3 src/main.py --predict "Your news text here" --model random_forest
```

#### 5. Interactive Mode
```bash
# Start interactive prediction mode
python3 src/main.py --interactive
```

#### 6. View Model Information
```bash
# Show all available models
python3 src/main.py --info

# Show specific model information
python3 src/main.py --info logistic
```

## Complete Workflow Examples

### Example 1: Complete Pipeline
```bash
# Step 1: Preprocess data and save it
python3 src/main.py --preprocess --save-data

# Step 2: Train and compare models
python3 src/main.py --compare --tune

# Step 3: Use interactive mode for predictions
python3 src/main.py --interactive
```

### Example 2: Quick Training and Prediction
```bash
# Train a model and make a prediction
python3 src/main.py --train logistic
python3 src/main.py --predict "Breaking news: Scientists discover new planet"
```

### Example 3: Deep Learning Models
```bash
# Train LSTM model
python3 src/main.py --train lstm

# Train CNN model
python3 src/main.py --train cnn

# Train Transformer model
python3 src/main.py --train transformer
```

## Command Line Options

### Main Actions
- `--preprocess`: Preprocess the data
- `--train MODEL_TYPE`: Train a specific model
- `--compare`: Compare multiple models
- `--predict TEXT`: Predict if text is fake news
- `--interactive`: Start interactive prediction mode
- `--info [MODEL_TYPE]`: Show model information

### Options
- `--tune`: Perform hyperparameter tuning
- `--save-data`: Save processed data to CSV
- `--model MODEL_TYPE`: Specify model for prediction
- `--data-dir PATH`: Data directory path (default: ../data)
- `--model-dir PATH`: Model directory path (default: ../model)

### Available Model Types
- **Traditional ML**: `logistic`, `svm`, `random_forest`, `naive_bayes`, `xgboost`
- **Deep Learning**: `lstm`, `bilstm`, `cnn`, `transformer`

## Interactive Mode

When you run `python3 src/main.py --interactive`, you can:

1. Enter news articles one by one
2. Get instant predictions with confidence scores
3. Type 'quit' to exit

Example session:
```
🔮 Interactive Prediction Mode
==================================================
Enter news articles to classify (type 'quit' to exit)

📰 Enter news article: Scientists discover new planet in our solar system
🎯 Prediction: True
📊 Confidence: 0.9234
📉 Fake Probability: 0.0766
📈 True Probability: 0.9234
🤖 Model Used: logistic
--------------------------------------------------

📰 Enter news article: quit
👋 Goodbye!
```

## File Structure

The system will create the following structure:
```
Fake News Detection/
├── data/
│   ├── fake.csv
│   ├── true.csv
│   └── processed_data.csv (if --save-data used)
├── model/
│   ├── *_model_*.pkl (trained models)
│   ├── *_metadata.json (model metadata)
│   └── session_*.json (session files)
└── src/
    └── main.py
```

## Tips

1. **First Time Setup**: Run `--preprocess` first to prepare your data
2. **Model Comparison**: Use `--compare` to find the best model for your data
3. **Hyperparameter Tuning**: Add `--tune` for better performance (slower)
4. **Interactive Mode**: Great for testing with real news articles
5. **Save Data**: Use `--save-data` to save preprocessed data for faster subsequent runs

## Troubleshooting

### Common Issues

1. **"No module named 'xgboost'"**: Install missing dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **"File not found"**: Make sure your CSV files are in the data directory

3. **Memory issues with deep learning**: Use traditional ML models first:
   ```bash
   python3 src/main.py --compare
   ```

4. **Slow training**: Skip hyperparameter tuning for faster results:
   ```bash
   python3 src/main.py --train logistic  # No --tune flag
   ```

### Getting Help

Run without arguments to see all available options:
```bash
python3 src/main.py
```

Or use the help flag:
```bash
python3 src/main.py --help
```
