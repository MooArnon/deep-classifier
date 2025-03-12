##########
# Import #
##############################################################################

import argparse
import os

import polars as pl

from deep_classifier.main import run_full_training_pipeline

###########
# Statics #
##############################################################################

control_column = ""
target_column = ""

#############
# Functions #
##############################################################################

def main(
        control_column: str,
        target_column: str,
        validation_split: float, 
        batch_size: int,
        epochs: int,
        max_trials: int,
) -> None:
    
    data = pl.read_csv(
        "data.csv"
    )
    
    run_full_training_pipeline(
        df=data,
        control_column=control_column,
        target_column=target_column,
        fe_methods=["percent_change_df", "rsi_df", "macd_df", "percent_price_ema_df"],
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        max_trials=max_trials,
    )

##############################################################################

def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_column", help="Stamp column")
    parser.add_argument("target_column", help="Taget price column")
    parser.add_argument("--validation-split", help="Number of test data splited", default=0.1)
    parser.add_argument("--batch-size", help="Batch size per epoch", default=32)
    parser.add_argument("--epochs", help="Number of epoch per trials", default=100)
    parser.add_argument("--max_trials", help="Search number", default=3)
    
    return parser.parse_args()

###########
# Running #
##############################################################################

if __name__ == "__main__":
    
    args = setup_args()
    main(
        control_column=args.control_column,
        target_column=args.target_column,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_trials=args.max_trials,
    )

##############################################################################
