from models.get_models import get_machine_learning_models, get_deep_learning_models


def train_models(X_tfidf_train, y_train, X_seq_train):
    """
    Trains and returns a dictionary of machine learning and deep learning models.
    """
    ml_models = get_machine_learning_models()
    dl_models = get_deep_learning_models()

    # Train machine learning models on TF-IDF features
    for name, model in ml_models.items():
        print(f"Training {name} model...")
        model.fit(X_tfidf_train, y_train)
        print(f"{name} model trained successfully.")

    # Train deep learning models on sequence data
    for name, model in dl_models.items():
        print(f"Training {name} model...")
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.fit(X_seq_train, y_train, epochs=20, batch_size=32, verbose=0)
        print(f"{name} model trained successfully.")

    print("All models trained successfully.")
    return {**ml_models, **dl_models}
