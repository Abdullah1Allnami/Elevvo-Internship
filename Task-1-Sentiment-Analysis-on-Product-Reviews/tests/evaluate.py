import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model


def evaluate_model(model, test_data, test_labels):
    """
    Evaluates the given model on the test data and returns the accuracy.

    Args:
        model: A trained scikit-learn or Keras model.
        test_data: The input data for testing.
        test_labels: The true labels for testing.

    Returns:
        float: Accuracy of the model on the test data.
    """
    model_name = model.name if isinstance(model, Model) else model.__class__.__name__
    print(f"Evaluating model: {model_name}")

    if hasattr(model, "predict"):
        if isinstance(model, Model):  # Keras model
            predictions = model.predict(test_data, verbose=0)

            # Convert probabilities to binary class labels (sigmoid output)
            if predictions.shape[-1] == 1:
                predictions = (predictions > 0.5).astype("int32").flatten()
            else:
                predictions = np.argmax(predictions, axis=-1)

        else:  # Scikit-learn model
            predictions = model.predict(test_data)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"{model_name} Accuracy: {accuracy:.2f}")
        return accuracy

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
