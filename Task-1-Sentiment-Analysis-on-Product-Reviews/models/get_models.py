from sklearn.linear_model import LogisticRegression
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
    BatchNormalization,
)


def get_machine_learning_models():
    """
    Returns a dictionary of machine learning models with their names as keys.
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": MultinomialNB(),
    }

    return models


def get_deep_learning_models(vocab_size=20000, max_len=500, embed_dim=128, num_heads=4):
    """
    Returns a dictionary of deep learning models with their names as keys.
    """
    models = {}

    # 1. Simple Feedforward
    models["Simple Feedforward"] = Sequential(
        [
            Dense(128, activation="relu", input_shape=(500,)),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    # 2. LSTM Model
    lstm_input = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(lstm_input)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)
    models["LSTM"] = Model(inputs=lstm_input, outputs=output)

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
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)
    models["Transformer-Attention"] = Model(inputs=trans_input, outputs=output)

    return models
