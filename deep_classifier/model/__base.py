##########
# Import #
##############################################################################

import os
from datetime import datetime, timezone
import pandas as pd
import pickle
import polars as pl

import numpy as np
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..fe.__base import BaseFE
from .custom_metric import BalancedLabelMetric

##############################################################################

class BaseModel:
    def __init__(self):
        pass

##############################################################################

class ModelWrapper:
    def __init__(
            self, 
            model: tf.keras.Model = None, 
            model_path: str = None,
            base_model_path: os.PathLike=None,
            fe_pipeline: BaseFE = None,
    ):
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
            Warning(
                "Either a Keras model or a model path must be provided."
            )
        
        if fe_pipeline:
            self.fe_pipeline = fe_pipeline
        else:
            Warning("Please assign fe_pipeline")
        
        if base_model_path:
            self.base_model_path = base_model_path
        else:
            Warning("Please assign fe_pipeline")
        
        self.model_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    ##########################################################################
    
    def build_fine_tune_model(self, hp):
        """
        Build a BTC-specific fine-tuned model by freezing base layers
        and adding dynamically-tuned top layers.
        """
        # Load base model and freeze its layers
        base_model = self.model
        base_model.trainable = False

        # Extract output from the second-to-last layer as a representation
        intermediate_output = base_model.layers[-2].output  

        x = intermediate_output

        # Dynamically choose number of fine-tuning layers
        num_layers = hp.Int('num_layers', min_value=1, max_value=5)
        for i in range(num_layers):
            units = hp.Int(f'ft_units_{i}', min_value=32, max_value=512, step=32)
            x = tf.keras.layers.Dense(units, activation='relu', name=f'ft_dense_{i+1}')(x)

        # Final binary classification layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='ft_output')(x)

        # Create the fine-tuned model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

        # Compile with dynamic learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='log')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[BalancedLabelMetric()],
        )

        return model

    ##########################################################################
    
    def fine_tune(
            self,
            df: pl.DataFrame,
            max_trials: int = 10,
            validation_split: float = 0.2,
            batch_size: int = 32,
            epochs: int = 50,
    ):
        """
        Fine-tune the model on BTC-specific data.

        Parameters
        ----------
        df : pl.DataFrame
            Raw BTC dataset.
        max_trials : int
            Number of hyperparameter tuning trials.
        validation_split : float
            Fraction of data for validation.
        batch_size : int
            Batch size.
        epochs : int
            Number of training epochs.

        Returns
        -------
        best_hps : dict
            The best hyperparameters found.
        val_loss : float
            The best validation loss achieved.
        best_model : tf.keras.Model
            The fine-tuned model.
        """

        if not self.fe_pipeline:
            raise ValueError("Feature engineering pipeline (fe_pipeline) is missing.")

        # 🔹 Apply Feature Engineering
        df_transformed = self.fe_pipeline.transform_df(df).drop_nulls()
        df_transformed.to_pandas().to_csv("transformed.csv")

        if df_transformed.height == 0:
            raise ValueError("No valid records after feature engineering.")

        # Prepare data
        X = df_transformed.drop(["label"]).to_numpy()
        y = df_transformed.select('label').to_numpy()
        
        # Count occurrences
        num_long = np.sum(y == 1)  # Count 1s (LONG)
        num_short = np.sum(y == 0)  # Count 0s (SHORT)
        print(f"LONG (1): {num_long}, SHORT (0): {num_short}")

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # 🔹 Setup Keras Tuner
        current_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        model_id = f"fine_tune_{current_timestamp}"
        tuner = kt.RandomSearch(
            hypermodel=self.build_fine_tune_model,
            objective=kt.Objective("val_balanced_label_metric", direction="max"),
            max_trials=max_trials,
            executions_per_trial=1,
            directory=model_id,
            project_name=model_id
        )

        # 🔹 Fine-Tune the Model
        tuner.search(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", 
                    patience=10, 
                    min_delta=0.0001,
                    restore_best_weights=True,
                )
            ],
            verbose=1
        )

        # Get best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate model
        val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)

        print("\nBest hyperparameters:", best_hps.values)
        y_pred = (best_model.predict(X_val) > 0.5).astype(int)
        print("\nClassification Report for Best Model:")
        print(classification_report(y_val, y_pred, target_names=['SHORT', 'LONG']))

        # 🔹 Save Fine-Tuned Model
        self.model = best_model
        return best_hps.values, val_loss, best_model

    ##########################################################################
    
    def predict(self, df: pl.DataFrame) -> str:
        """
        Apply feature engineering and predict binary labels.

        Parameters
        ----------
        df : pl.DataFrame
            Raw BTC data.
        threshold : float, optional
            Classification threshold.

        Returns
        -------
        df : pl.DataFrame
            DataFrame with predictions.
        """
        # Apply feature engineering
        df_transformed = self.fe_pipeline.transform_df(df)

        # Prepare data for prediction
        X = df_transformed.drop(["label"]).to_numpy()

        # Perform model inference
        probs: float = self.model.predict(X)
        predictions = probs.round().astype(int)

        # Convert to readable labels
        labels = ["LONG" if p == 1 else "SHORT" for p in predictions]

        # Add predictions to DataFrame
        df_transformed = df_transformed.with_columns(
            pl.Series("predicted_label", labels)
        )

        return df_transformed
    
    ##########################################################################
    
    def predict_one(self, data: dict) -> str:
        """
        Apply feature engineering and predict binary labels.

        Parameters
        ----------
        df : pl.DataFrame
            Raw BTC data.
        threshold : float, optional
            Classification threshold.

        Returns
        -------
        df : pl.DataFrame
            DataFrame with predictions.
        """
        pd_df = pd.DataFrame(data)
        df = pl.from_pandas(pd_df)
        
        # Apply feature engineering
        df_transformed = self.fe_pipeline.transform_df(df)
        print(df_transformed.columns)
        
        # Sort by control column (timestamp) to ensure we get the latest row
        # df_transformed = df_transformed.sort(self.fe_pipeline.control_column)

        # Select the most recent row
        latest_features = df_transformed[-1, :].drop(["label"]).to_numpy()

        # Perform model inference
        probs = self.model.predict(latest_features.reshape(1, -1))
        predictions = probs[-1][0].round().astype(int)

        return predictions

    ##########################################################################
    
    def save(self, filepath: str):
        """
        Save the entire ModelWrapper (FE + Model) as a single pickle file.

        Parameters
        ----------
        filepath : str
            Filepath to save the wrapper.
        """
        keras_model_path = f"{filepath}.h5"
        self.model.save(keras_model_path)  # Save Keras model separately

        # Temporarily remove the model from the object to save as pickle
        model_backup = self.model
        self.model = None

        with open(f"{filepath}_wrapper.pkl", 'wb') as file:
            pickle.dump(self, file)

        # Restore model reference
        self.model = model_backup
    
    ##########################################################################
    
    @staticmethod
    def load(path: os.PathLike ,filepath: str = 'model_base'):
        """
        Load a ModelWrapper object from a pickle file.

        Parameters
        ----------
        filepath : str
            Base filename without extension.

        Returns
        -------
        ModelWrapper
            Loaded ModelWrapper instance.
        """
        wrapper_path = os.path.join(path, f"{filepath}_wrapper.pkl")
        model_path = os.path.join(path, f"{filepath}.h5")
        
        with open(wrapper_path, 'rb') as file:
            wrapper = pickle.load(file)

        wrapper.model = tf.keras.models.load_model(model_path)
        return wrapper
    
    ##########################################################################
    
    @staticmethod
    def load_tuned(path: os.PathLike ,filepath: str):
        """
        Load a ModelWrapper object from a pickle file.

        Parameters
        ----------
        filepath : str
            Base filename without extension.

        Returns
        -------
        ModelWrapper
            Loaded ModelWrapper instance.
        """
        wrapper_path = os.path.join(path, f"{filepath}_wrapper.pkl")
        model_path = os.path.join(path, f"{filepath}.h5")
        
        with open(wrapper_path, 'rb') as file:
            wrapper = pickle.load(file)

        wrapper.model = tf.keras.models.load_model(model_path)
        return wrapper
    
    ##########################################################################
    
##############################################################################