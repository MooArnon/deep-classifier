##########
# Import #
##############################################################################

import pickle
from datetime import datetime, timezone
import os 
import logging 
from typing import Union

import pandas as pd
import numpy as np
import polars as pl
from space_time_pipeline.data_lake_house import Athena
from space_time_pipeline.data_lake import S3DataLake
import tensorflow as tf
from space_time_pipeline.nosql.dynamo_db import DynamoDB

from deep_classifier.fe.classifier_fe import ClassifierFE
from deep_classifier.model.trainer import Trainer
from deep_classifier.model.__base import ModelWrapper
from deep_classifier.utilities.s3 import clear_and_push_to_s3, upload_to_s3

##############
# Frameworks #
##############################################################################

def run_full_training_pipeline(
        logger: logging.Logger,
        control_column: str = "date",
        target_column: str = "close",
        fe_methods: list = ["percent_change_df", "rsi_df", "macd_df", "percent_price_ema_df"],
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        max_trials: int = 10,
        push_to_s3: bool = False,
        file_path: os.PathLike = os.path.join(''),
        model_base_wrapper_name: str = "model_base_wrapper.pkl",
        model_base_name: str = "model_base.h5",
        aws_s3_bucket: str = "space-time-model",
        aws_s3_prefix: str = "classifier",
        percent_change_window: list[int] = [1, 2, 3, 4, 5, 9, 22, 30],
        percent_price_ema_window: list[int] = [7, 22, 99]
        
):  
    # Set GPU memory growth to avoid memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(
                f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs"
            )
        except RuntimeError as e:
            logger.info(e)

    # Ensure TensorFlow runs on GPU
    with tf.device('/GPU:0'):
        logger.info("Running on GPU!")
        
    model_base_wrapper_path = os.path.join(file_path, model_base_wrapper_name)
    model_base_path = os.path.join(file_path, model_base_name)
    
    aws_s3_prefix = f"{aws_s3_prefix}/base"
    
    # logger.info(f"Selecting data from {os.path.join('sql', 'select_all_data.sql')}")
    # __get_data_athena(
    #     logger=logger,
    #     query_file_path=os.path.join('sql', 'select_all_data.sql'),
    #     replace_condition_dict={'<LIMIT>': 10000000},
    #     file_name='data-all.csv'
    # )
    
    df = pl.read_csv(
        os.path.join(
            'local',
            'all-data.csv'
        ),
        infer_schema_length=10000,
        schema_overrides={
            "volume": pl.Float64,
            "taker_buy_base_asset_volume": pl.Float64,
        },
    )
    logger.info(f"Train with shape: {df.shape}")
    
    # Initialize feature engineering
    fe = ClassifierFE(
        control_column=control_column,
        target_column=target_column,
        fe_name_list=fe_methods,
        percent_change_window=percent_change_window,
        percent_price_ema_window=percent_price_ema_window,
    )

    # Transform features
    df_features = fe.transform_df(df)
    logger.info(f"Features: {df_features.columns}")

    # Prepare data for model
    X = df_features.drop(["label"]).to_numpy()
    y = df_features.select('label').to_numpy()
    
    # Count occurrences
    num_long = np.sum(y == 1)  # Count 1s (LONG)
    num_short = np.sum(y == 0)  # Count 0s (SHORT)
    logger.info(f"LONG (1): {num_long}, SHORT (0): {num_short}")

    logger.info(f"TensorFlow version: {tf.__version__}")
    devices = tf.config.list_physical_devices()
    logger.info(f"All physical devices: {devices}")

    gpu_devices = tf.config.list_physical_devices("GPU")
    logger.info(f"GPU devices: {gpu_devices}")

    # Train the model with random search
    trainer = Trainer(model_type='dnn', input_shape=(X.shape[1],))

    best_params, val_loss, best_model = trainer.random_search(
        X, y,
        max_trials=max_trials,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs
    )
    logger.info(f"Got best model with arch: {best_model.summary()}")

    # Wrap and save the best model
    wrapper = ModelWrapper(model=best_model, fe_pipeline=fe)
    wrapper.save('model_base')
    
    # Push to S3
    if push_to_s3:
        logger.info("Push model and wrapper to S3")
        
        # Wrapper 
        clear_and_push_to_s3(
            model_base_wrapper_path,
            aws_s3_bucket,
            f"{aws_s3_prefix}/",
        )
        logger.info(
            f"Pushed {model_base_wrapper_path} to " \
                f"bucket: {aws_s3_bucket} prefix: {aws_s3_prefix}"
        )
        
        # Keras model
        upload_to_s3(
            model_base_path,
            aws_s3_bucket,
            f"{aws_s3_prefix}/{model_base_name}",
        )
        logger.info(
            f"Pushed {model_base_path} to " \
                f"bucket: {aws_s3_bucket} prefix: {aws_s3_prefix}"
        )
        
        os.remove(model_base_wrapper_path)
        os.remove(model_base_path)

    return best_params, val_loss, best_model

##############################################################################

def run_fine_tune_pipeline(
        logger: logging.Logger,
        base_model_path: str,
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 50,
        max_trials: int = 3,
        asset: str = 'btc',
        aws_s3_bucket: str = "space-time-model",
        aws_s3_prefix: str = "classifier",
        push_to_s3: bool = False,
        model_fine_tuned_wrapper_name: str = "fine_tuned_model_wrapper.pkl",
        model_fine_tuned_name: str = "fine_tuned_model.h5",
):
    """
    Fine-tune a pre-trained model using BTC data.

    Parameters
    ----------
    base_model_path : str
        Path to the saved base model.
    control_column : str
        The column to use for sorting and time-based transformations.
    target_column : str
        The column to predict.
    fe_methods : list
        List of feature engineering methods.
    validation_split : float
        Fraction of data for validation.
    batch_size : int
        Batch size.
    epochs : int
        Number of epochs.
    max_trials : int
        Number of hyperparameter tuning trials.
    asset : str
        Asset name (default is 'btc').

    Returns
    -------
    best_params : dict
        The best hyperparameters found.
    val_loss : float
        The best validation loss achieved.
    best_model : tf.keras.Model
        The fine-tuned model.
    """
    local_path="base_model"
    
    # Local path
    base_model_path = os.path.join(local_path, base_model_path)
    
    # S3 prefix
    aws_s3_prefix_base = f"{aws_s3_prefix}/base"
    
    s3 = S3DataLake(logger=logger)
    s3.download_file(
        bucket_name=aws_s3_bucket,
        target_prefix=aws_s3_prefix_base,
        logger=logger,
        local_path=local_path,
    )
    
    tmp_wrapper =  ModelWrapper()
    wrapper: ModelWrapper = tmp_wrapper.load(
        path=local_path
    )
    
    # 🔹 Step 1: Fetch Data
    # __get_data_athena(
    #     logger=logger,
    #     query_file_path=os.path.join('sql', 'select_btc_data.sql'),
    #     replace_condition_dict={'<LIMIT>': 1000000},
    #     file_name=f'data-{asset}.csv'
    # )
    
    df = pl.read_csv(
        os.path.join(
            "local",
            "btc-data.csv",
        )
    )
    if df.columns[0] == "":
        df = df.drop(df.columns[0])
    logger.info(f"Loaded Data Shape: {df.shape}")
    
    # 🔹 Step 4: Fine-Tune Model
    best_params, val_loss, best_model = wrapper.fine_tune(
        df,
        max_trials=max_trials,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # 🔹 Step 5: Save Fine-Tuned Model
    wrapper.save(f"{asset}_fine_tuned_model")

    print(best_model.summary())
    
    # Push to S3
    if push_to_s3:
        aws_s3_prefix=f"{aws_s3_prefix}/tuned/{asset}"
        model_fine_tuned_wrapper_name = f"{asset}_{model_fine_tuned_wrapper_name}"
        model_fine_tuned_name = f"{asset}_{model_fine_tuned_name}"
        
        model_fine_tuned_wrapper_path = os.path.join(model_fine_tuned_wrapper_name)
        model_fine_tuned_path = os.path.join(model_fine_tuned_name)
        
        print(f"model_fine_tuned_wrapper_path: {model_fine_tuned_wrapper_path}")

        logger.info("Push model and wrapper to S3")
        
        # Wrapper 
        clear_and_push_to_s3(
            model_fine_tuned_wrapper_path,
            aws_s3_bucket,
            f"{aws_s3_prefix}/",
        )
        logger.info(
            f"Pushed {model_fine_tuned_wrapper_path} to " \
                f"bucket: {aws_s3_bucket} prefix: {aws_s3_prefix}"
        )
        
        # Keras model
        upload_to_s3(
            model_fine_tuned_path,
            aws_s3_bucket,
            f"{aws_s3_prefix}/{model_fine_tuned_name}",
        )
        logger.info(
            f"Pushed {model_fine_tuned_path} to " \
                f"bucket: {aws_s3_bucket} prefix: {aws_s3_prefix}"
        )
        
        os.remove(model_fine_tuned_wrapper_path)
        os.remove(model_fine_tuned_path)

    return best_params, val_loss, best_model

##############################################################################

def predict(
        asset: str, 
        logger: logging.Logger, 
        model_path: os.PathLike = "",
        model_name: str = "fine_tuned_model",
) -> Union[int, str]:
    model_name = f"{asset}_{model_name}"

    wrapper: ModelWrapper = ModelWrapper.load_tuned(
        path = model_path,
        filepath=model_name
    )
    
    logger.info(f"Predict with features: {wrapper.fe_pipeline.features}")
    logger.info("Model architecture:")
    logger.info(f"{wrapper.model.summary()}")
    logger.info(f"Model id is: {wrapper.model_id}")
    
    dynamo = DynamoDB(logger)
    data = dynamo.print_all_records('price_stream')
    
    return wrapper.predict_one(data), wrapper.model_id

##############################################################################
