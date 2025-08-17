from models.get_models import get_machine_learning_models, get_deep_learning_models
import tensorflow as tf
from tests.evaluate_model import evaluate_model
from transformers import AutoTokenizer


def train_and_evaluate_models(
    X_tfidf_train, X_tfidf_test, X_seq_train, X_seq_test, y_train, y_test, num_classes
):
    """
    Trains and returns a dictionary of machine learning and deep learning models.
    """

    ml_models = get_machine_learning_models()
    dl_models = get_deep_learning_models(num_classes=num_classes)
    accuracy_results = {}

    # Train machine learning models on TF-IDF features
    for name, model in ml_models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_tfidf_train, y_train)
        print(f"{name} model training completed.")
        print(f"Training Accuracy: {model.score(X_tfidf_train, y_train):.4f}")

        # Evaluate training accuracy
        accuracy = model.score(X_tfidf_test, y_test)
        print(f"{name} Evaluation Accuracy: {accuracy:.4f}")
        accuracy_results[name] = accuracy

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    # Train deep learning models on sequence data
    for name, model in dl_models.items():
        print(f"\nTraining {name} model...")
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # if the saved_models directory does not exist, create it
        tf.io.gfile.makedirs("./saved_models")

        # Use ModelCheckpoint to save the best model during training
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./saved_models/{name}_best_model.h5",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        )

        # Add verbose output for deep learning models
        history = model.fit(
            X_seq_train,
            y_train_onehot,
            validation_split=0.1,
            epochs=10,
            callbacks=[checkpoint_callback],
            batch_size=32,
            verbose=1,  # Set verbose=1 to print progress for each epoch
        )
        print(f"{name} model training completed.")
        # Print training accuracy for each epoch
        for epoch, acc in enumerate(history.history["accuracy"], 1):
            print(f"Epoch {epoch}: Training Accuracy = {acc:.4f}")

        # Evaluate training accuracy
        accuracy = evaluate_model(model, X_seq_test, y_test)
        print(f"{name} Evaluation Accuracy: {accuracy:.4f}")
        accuracy_results[name] = accuracy

    print("Evaluation Accuracy Results:", accuracy_results)

    return {**ml_models, **dl_models}
