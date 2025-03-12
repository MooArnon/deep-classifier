##########
# Import #
##############################################################################

from datetime import datetime
import pickle
from typing import Union

import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from .__base import ModelWrapper
from deep_classifier.model.classifier_model import build_dynamic_dnn_hp

###########
# Classes #
##############################################################################

class Trainer:
    def __init__(
            self, 
            model_type: str, 
            input_shape: tuple,
    ):
        """
        Initialize the Trainer with a function that builds a compiled model.
        
        Parameters
        ----------
        build_model_func : callable
            A function that returns a compiled tf.keras.Model.
            It should accept hyperparameter values as keyword arguments.
        """
        self.model = None
        self.input_shape = input_shape
        self.model_type = model_type
        
        if model_type == 'dnn':
            self.build_model_func = build_dynamic_dnn_hp

    ##########################################################################
    
    def train(self, X, y, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the model.
        
        Parameters
        ----------
        X : np.ndarray or similar
            Training features.
        y : np.ndarray or similar
            Training labels.
        epochs : int, optional
            Number of epochs.
        batch_size : int, optional
            Batch size.
        validation_split : float, optional
            Fraction of data to use for validation.
        
        Returns
        -------
        history : History object
            The training history.
        """
        # Build a new model.
        self.model = self.build_model_func()
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[es]
        )
        return history

    ##########################################################################
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the provided data.
        
        Parameters
        ----------
        X : np.ndarray or similar
            Evaluation features.
        y : np.ndarray or similar
            Evaluation labels.
        
        Returns
        -------
        tuple
            Loss and metrics as returned by model.evaluate.
        """
        return self.model.evaluate(X, y)

    ##########################################################################
    
    def random_search(
            self, 
            X, 
            y, 
            max_trials: int = 10, 
            validation_split: float = 0.2,
            batch_size: int = 32, 
            epochs: int = 50,
    ) -> Union[dict, float, Model]:
        model_id = datetime.today().strftime('%Y%m%d__%H%M%S')
        tuner = kt.RandomSearch(
            hypermodel=lambda hp: self.build_model_func(hp, self.input_shape),
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=1,
            directory=f'random_search_{model_id}',
            project_name=f'random_search_{model_id}'
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        tuner.search(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", 
                    patience=5, 
                    restore_best_weights=True,
                )
            ]
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate best model
        val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)

        print("Best hyperparameters:", best_hps.values)
        y_pred = (best_model.predict(X_val) > 0.5).astype(int)
        print("\nClassification Report for Best Model:")
        print(classification_report(y_val, y_pred, target_names=['SHORT', 'LONG']))
        
        wrapper = ModelWrapper(best_model)
        wrapper.save(self.model_type)

        return best_hps.values, val_loss, best_model
    
    ##########################################################################
    
##############################################################################