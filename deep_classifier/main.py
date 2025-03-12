##########
# Import #
##############################################################################

import os 

import polars as pl

from deep_classifier.fe.classifier_fe import ClassifierFE
from deep_classifier.model.trainer import Trainer
from deep_classifier.model.__base import ModelWrapper

##############
# Frameworks #
##############################################################################

def run_full_training_pipeline(
        df: pl.DataFrame,
        control_column: str = "date",
        target_column: str = "close",
        fe_methods: list = ["percent_change_df", "rsi_df", "macd_df", "percent_price_ema_df"],
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        max_trials: int = 10,
):
    # Initialize feature engineering
    fe = ClassifierFE(
        control_column=control_column,
        target_column=target_column,
        fe_name_list=fe_methods
    )

    # Transform features
    df_features = fe.transform_df(df)

    df_features.to_pandas().to_csv("transformed.csv")

    # Prepare data for model
    X = df_features.drop(["label"]).to_numpy()
    y = (df_features["label"] == "LONG").to_numpy().astype(int)

    # Train the model with random search
    trainer = Trainer(model_type='dnn', input_shape=(X.shape[1],))

    best_params, val_loss, best_model = trainer.random_search(
        X, y,
        max_trials=max_trials,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs
    )

    # Wrap and save the best model
    wrapper = ModelWrapper(best_model)
    wrapper.save('model_final')
    
    print(best_model.summary())

    return best_params, val_loss, best_model

##############################################################################
