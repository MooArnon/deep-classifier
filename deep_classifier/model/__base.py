##########
# Import #
##############################################################################

import os
import pickle

import tensorflow as tf

##############################################################################

class BaseModel:
    def __init__(self):
        pass

##############################################################################

class ModelWrapper:
    def __init__(self, model: tf.keras.Model = None, model_path: str = None):
        """
        Wrap the trained Keras model for inference.

        Parameters
        ----------
        model : tf.keras.Model
            A trained Keras model.
        model_path : str
            Path to load the model (.h5 file).
        """
        if model:
            self.model = model
        elif model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Either a Keras model or a model path must be provided.")

    def predict(self, X, threshold: float = 0.5):
        """
        Predict binary labels.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        threshold : float
            Threshold to classify probabilities into binary labels.

        Returns
        -------
        np.ndarray
            Binary predictions.
        """
        probs = self.model.predict(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str):
        """
        Save only the Keras model (.h5 file).

        Parameters
        ----------
        filepath : str
            Filepath including the .h5 extension.
        """
        self.model.save(f"{filepath}.h5", include_optimizer=False)