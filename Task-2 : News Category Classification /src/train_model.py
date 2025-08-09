from models.get_models import get_machine_learning_models, get_deep_learning_models


def train_models(X_tfidf_train, y_train, X_seq_train):
    """
    Trains and returns a dictionary of machine learning and deep learning models.
    """
    ml_models = get_machine_learning_models()
    dl_models = get_deep_learning_models()

    # Train machine learning models on TF-IDF features
    for name, model in ml_models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_tfidf_train, y_train)
        print(f"{name} model training completed.")

    # Train deep learning models on sequence data
    for name, model in dl_models.items():
        print(f"\nTraining {name} model...")
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        # Add verbose output for deep learning models
        history = model.fit(
            X_seq_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=1,  # Set verbose=1 to print progress for each epoch
        )
        print(f"{name} model training completed.")
        # Print training accuracy for each epoch
        for epoch, acc in enumerate(history.history["accuracy"], 1):
            print(f"Epoch {epoch}: Training Accuracy = {acc:.4f}")

    print("\nAll models trained successfully.")
    return {**ml_models, **dl_models}
