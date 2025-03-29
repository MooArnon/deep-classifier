import pytest
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from deep_classifier.model.classifier_model import build_dynamic_dnn_hp  # Adjust import path as needed
from deep_classifier.model.trainer import Trainer
from tensorflow.keras.models import Model
from deep_classifier.model.__base import ModelWrapper

def test_build_dynamic_dnn_hp():
    # Mock the hyperparameter object from Keras Tuner
    hp = kt.HyperParameters()

    # Define hyperparameters explicitly
    hp.Fixed('num_layers', 3)
    hp.Fixed('learning_rate', 1e-3)

    # Define units per layer explicitly for consistency
    for i in range(3):
        hp.Fixed(f'dense_units_{i}', 64 * (i + 1))

    input_shape = (20,)  # Example feature size

    model = build_dynamic_dnn_hp(hp, input_shape)

    # Assert the output
    assert isinstance(model, Model), "The returned object must be a Keras Model instance."

    # Check number of Dense layers
    dense_layers = [layer for layer in model.layers if "dense" in layer.name]
    expected_layers = 4  # 3 hidden layers + 1 output layer
    assert len(dense_layers) == expected_layers, f"Expected {expected_layers} dense layers, but got {len(dense_layers)}."

    # Check compilation
    assert model.loss == 'binary_crossentropy', "Loss function should be binary_crossentropy."
    assert isinstance(model.optimizer, type(model.optimizer)), "Optimizer should be properly instantiated."
    
@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.rand(500, 10)
    y = np.random.randint(0, 2, 500)
    return X, y

def test_trainer_random_search(dummy_data, tmp_path):
    X, y = dummy_data
    input_shape = (X.shape[1],)

    trainer = Trainer(model_type='dnn', input_shape=input_shape)
    
    best_params, val_loss, best_model = trainer.random_search(
        X, y,
        max_trials=2,  # For speed, keep small
        epochs=10,
        validation_split=0.2,
        batch_size=16
    )

    # Check best_params is a dict and contains expected keys
    assert isinstance(best_params, dict), "best_params should be a dictionary."
    assert "num_layers" in best_params, "num_layers not found in best_params."
    assert "learning_rate" in best_params, "learning_rate not found in best_params."

    # Check best_model is a Keras model
    assert hasattr(best_model, 'predict'), "best_model should have a predict method."

    # Check val_loss is a float
    assert isinstance(val_loss, float), "val_loss should be a float."

    # Save model wrapper temporarily
    model_path = tmp_path / "test_model"
    wrapper = ModelWrapper(best_model)
    wrapper.save(str(model_path))

    # Verify the file was created
    saved_model_file = f"{model_path}.h5"
    assert os.path.exists(saved_model_file), f"Model file {saved_model_file} was not created."

    # Load model using wrapper and test prediction
    loaded_wrapper = ModelWrapper(model_path=str(saved_model_file))
    predictions = loaded_wrapper.predict(X[:10])
    
    # Check predictions shape and values
    assert predictions.shape == (10, 1), "Predictions should have shape (10, 1)."
    assert np.isin(predictions, [0, 1]).all(), "Predictions should only contain binary values."