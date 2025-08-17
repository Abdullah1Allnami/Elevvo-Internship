import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model


def evaluate_model(model, X_seq_test, y_test):
    """
    Evaluates the given model on the test data and returns the accuracy.
    """
    model_name = model.name if isinstance(model, Model) else model.__class__.__name__
    print(f"\nEvaluating model: {model_name}")
    predictions = model.predict(X_seq_test)
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # For multi-class classification, take the class with the highest probability
        predictions = np.argmax(predictions, axis=1)
    else:
        # For binary classification, predictions are already in the correct format
        predictions = (predictions > 0.5).astype(int)

    test_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    accuracy = accuracy_score(test_labels, predictions)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    return accuracy
