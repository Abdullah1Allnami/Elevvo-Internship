from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


# Check for optional dependencies (lazy loading)
def check_xgboost():
    try:
        import xgboost

        return True
    except ImportError:
        return False
    except Exception:
        # Handle XGBoost installation issues (like missing OpenMP)
        return False


def check_tensorflow():
    try:
        import tensorflow as tf

        # Test basic functionality
        tf.constant([1, 2, 3])
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: TensorFlow has issues: {str(e)}")
        return False


# Don't check at import time - check when needed
XGBOOST_AVAILABLE = None
TENSORFLOW_AVAILABLE = None

# TensorFlow imports will be done inside functions when needed


def get_model(
    model_type, num_classes=2, input_dim=None, max_features=None, max_length=None
):
    """
    Get different types of models for fake news detection

    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of classes (default: 2 for binary classification)
        input_dim (int): Input dimension for neural networks
        max_features (int): Maximum number of features for embedding
        max_length (int): Maximum sequence length for neural networks

    Returns:
        model: The specified model
    """

    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver="liblinear"
        )

    elif model_type == "svm":
        model = SVC(kernel="linear", C=1.0, random_state=42, probability=True)

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
        )

    elif model_type == "naive_bayes":
        model = MultinomialNB(alpha=1.0)

    elif model_type == "xgboost":
        if not check_xgboost():
            raise ImportError(
                "XGBoost is not available. Please install it with: pip install xgboost"
            )
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    elif model_type == "lstm":
        if not check_tensorflow():
            raise ImportError(
                "TensorFlow is not available or has issues. Please install it with: pip install tensorflow"
            )
        if input_dim is None or max_features is None or max_length is None:
            raise ValueError(
                "For LSTM model, input_dim, max_features, and max_length must be provided"
            )

        try:
            import tensorflow as tf
            from tensorflow import keras

            model = keras.Sequential(
                [
                    keras.layers.Embedding(max_features, 128, input_length=max_length),
                    keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                    keras.layers.Dense(32, activation="relu"),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        except Exception as e:
            raise ImportError(
                f"Failed to create LSTM model due to TensorFlow issues: {str(e)}"
            )

    elif model_type == "bilstm":
        if not check_tensorflow():
            raise ImportError(
                "TensorFlow is not available. Please install it with: pip install tensorflow"
            )
        if input_dim is None or max_features is None or max_length is None:
            raise ValueError(
                "For BiLSTM model, input_dim, max_features, and max_length must be provided"
            )

        try:
            import tensorflow as tf
            from tensorflow import keras

            model = keras.Sequential(
                [
                    keras.layers.Embedding(max_features, 128, input_length=max_length),
                    keras.layers.Bidirectional(
                        keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
                    ),
                    keras.layers.Dense(32, activation="relu"),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        except Exception as e:
            raise ImportError(
                f"Failed to create BiLSTM model due to TensorFlow issues: {str(e)}"
            )

    elif model_type == "cnn":
        if not check_tensorflow():
            raise ImportError(
                "TensorFlow is not available. Please install it with: pip install tensorflow"
            )
        if input_dim is None or max_features is None or max_length is None:
            raise ValueError(
                "For CNN model, input_dim, max_features, and max_length must be provided"
            )

        try:
            import tensorflow as tf
            from tensorflow import keras

            model = keras.Sequential(
                [
                    keras.layers.Embedding(max_features, 128, input_length=max_length),
                    keras.layers.Conv1D(128, 5, activation="relu"),
                    keras.layers.MaxPooling1D(5),
                    keras.layers.Conv1D(128, 5, activation="relu"),
                    keras.layers.MaxPooling1D(5),
                    keras.layers.Conv1D(128, 5, activation="relu"),
                    keras.layers.GlobalAveragePooling1D(),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        except Exception as e:
            raise ImportError(
                f"Failed to create CNN model due to TensorFlow issues: {str(e)}"
            )

    elif model_type == "transformer":
        if not check_tensorflow():
            raise ImportError(
                "TensorFlow is not available. Please install it with: pip install tensorflow"
            )
        if input_dim is None or max_features is None or max_length is None:
            raise ValueError(
                "For Transformer model, input_dim, max_features, and max_length must be provided"
            )

        try:
            import tensorflow as tf
            from tensorflow import keras

            # Input layer
            inputs = keras.layers.Input(shape=(max_length,))

            # Embedding layer
            embedding = keras.layers.Embedding(max_features, 128)(inputs)

            # Multi-head attention
            attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=128)(
                embedding, embedding
            )
            attention = keras.layers.LayerNormalization()(attention)

            # Add & Norm
            attention = keras.layers.Add()([embedding, attention])
            attention = keras.layers.LayerNormalization()(attention)

            # Global average pooling
            pooled = keras.layers.GlobalAveragePooling1D()(attention)

            # Dense layers
            dense = keras.layers.Dense(128, activation="relu")(pooled)
            dense = keras.layers.Dropout(0.5)(dense)
            dense = keras.layers.Dense(64, activation="relu")(dense)
            dense = keras.layers.Dropout(0.3)(dense)
            outputs = keras.layers.Dense(num_classes, activation="softmax")(dense)

            model = keras.Model(inputs, outputs)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        except Exception as e:
            raise ImportError(
                f"Failed to create Transformer model due to TensorFlow issues: {str(e)}"
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def get_model_info(model_type):
    """
    Get information about different model types

    Args:
        model_type (str): Type of model

    Returns:
        dict: Model information
    """
    model_info = {
        "logistic": {
            "name": "Logistic Regression",
            "type": "Traditional ML",
            "description": "Linear classifier using logistic function",
            "pros": ["Fast", "Interpretable", "Good baseline"],
            "cons": ["Linear decision boundary", "May underfit complex patterns"],
        },
        "svm": {
            "name": "Support Vector Machine",
            "type": "Traditional ML",
            "description": "Finds optimal hyperplane for classification",
            "pros": ["Effective in high dimensions", "Memory efficient"],
            "cons": ["Slow on large datasets", "Sensitive to feature scaling"],
        },
        "random_forest": {
            "name": "Random Forest",
            "type": "Ensemble ML",
            "description": "Ensemble of decision trees",
            "pros": [
                "Handles non-linear patterns",
                "Feature importance",
                "Robust to overfitting",
            ],
            "cons": ["Can be slow", "Less interpretable than single tree"],
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "type": "Traditional ML",
            "description": "Probabilistic classifier based on Bayes theorem",
            "pros": [
                "Fast",
                "Good for text classification",
                "Works well with small datasets",
            ],
            "cons": [
                "Assumes feature independence",
                "Can be outperformed by other methods",
            ],
        },
        "xgboost": {
            "name": "XGBoost",
            "type": "Gradient Boosting",
            "description": "Gradient boosting framework",
            "pros": [
                "High performance",
                "Handles missing values",
                "Feature importance",
            ],
            "cons": ["Can overfit", "Requires parameter tuning", "Less interpretable"],
        },
        "lstm": {
            "name": "LSTM",
            "type": "Deep Learning",
            "description": "Long Short-Term Memory neural network",
            "pros": [
                "Captures sequential patterns",
                "Good for text",
                "Handles long sequences",
            ],
            "cons": ["Slow training", "Requires more data", "Black box"],
        },
        "bilstm": {
            "name": "Bidirectional LSTM",
            "type": "Deep Learning",
            "description": "LSTM that processes sequences in both directions",
            "pros": ["Better context understanding", "Improved performance"],
            "cons": ["Slower than unidirectional", "More parameters"],
        },
        "cnn": {
            "name": "Convolutional Neural Network",
            "type": "Deep Learning",
            "description": "CNN for text classification using 1D convolutions",
            "pros": ["Fast training", "Good for local patterns", "Parallelizable"],
            "cons": ["Limited context window", "May miss long-range dependencies"],
        },
        "transformer": {
            "name": "Transformer",
            "type": "Deep Learning",
            "description": "Attention-based neural network",
            "pros": [
                "Captures long-range dependencies",
                "State-of-the-art performance",
            ],
            "cons": ["Complex", "Requires large datasets", "Computationally expensive"],
        },
    }

    return model_info.get(
        model_type,
        {"name": "Unknown", "type": "Unknown", "description": "Unknown model type"},
    )


def compare_models():
    """
    Print comparison of all available models
    ry"""
    print("Available Models for Fake News Detection:")
    print("=" * 50)

    model_types = [
        "logistic",
        "svm",
        "random_forest",
        "naive_bayes",
        "xgboost",
        "lstm",
        "bilstm",
        "cnn",
        "transformer",
    ]

    for model_type in model_types:
        info = get_model_info(model_type)
        print(f"\n{info['name']} ({model_type})")
        print(f"Type: {info['type']}")
        print(f"Description: {info['description']}")
        print(f"Pros: {', '.join(info['pros'])}")
        print(f"Cons: {', '.join(info['cons'])}")
        print("-" * 30)


if __name__ == "__main__":
    compare_models()
