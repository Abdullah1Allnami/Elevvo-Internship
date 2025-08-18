from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    GRU,
    Input,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    Flatten,
    concatenate,
    Layer,
)
from transformers import TFAutoModel
import tensorflow as tf


def get_machine_learning_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(),
    }

    return models


def get_deep_learning_models(vocab_size=20000, max_len=500, embed_dim=128, num_heads=4):
    """
    Returns a dictionary of deep learning models with their names as keys.
    """
    models = {}
    models["BERT"] = get_pretrained_bert_model(num_classes)
    models["Enhanced Transformer"] = get_enhanced_transformer_model(num_classes)
    models["Hypered Model"] = get_hypered_model(num_classes)

    if num_classes is None:
        raise ValueError("num_classes must be specified for deep learning models")

    # 1. Simple Feedforward
    models["Simple Feedforward"] = Sequential(
        [
            Dense(256, activation="relu", input_shape=(max_len,)),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    # 2. LSTM Model
    lstm_input = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(lstm_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    models["Stacked LSTM"] = Model(inputs=lstm_input, outputs=output)

    # 3. GRU Model
    gru_input = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(gru_input)
    x = GRU(64)(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)
    models["GRU"] = Model(inputs=gru_input, outputs=output)

    # 4. Transformer-like Model (simple attention block)
    trans_input = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(trans_input)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    models["Transformer"] = Model(inputs=trans_input, outputs=output)

    return models
