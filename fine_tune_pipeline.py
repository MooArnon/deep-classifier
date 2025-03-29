##########
# Import #
##############################################################################

import argparse
import os

import polars as pl

from deep_classifier.utilities.logger import get_logger
from deep_classifier.main import run_fine_tune_pipeline

###########
# Statics #
##############################################################################

logger = get_logger(logger_name=os.path.basename(__file__))

#############
# Functions #
##############################################################################

def main(
        validation_split: float, 
        batch_size: int,
        epochs: int,
        max_trials: int,
) -> None:

    run_fine_tune_pipeline(
        base_model_path='model_base.h5',
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        max_trials=max_trials,
        logger=logger,
        push_to_s3=True,
    )

##############################################################################

def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-split", help="Number of test data splited", default=0.1, type = float)
    parser.add_argument("--batch-size", help="Batch size per epoch", default=32, type=int)
    parser.add_argument("--epochs", help="Number of epoch per trials", default=100, type=int)
    parser.add_argument("--max_trials", help="Search number", default=3, type=int)
    
    return parser.parse_args()

###########
# Running #
##############################################################################

if __name__ == "__main__":
    # python fine_tune_pipeline.py --epochs 150 --max_trials 10 --validation-split 0.2
    args = setup_args()
    main(
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_trials=args.max_trials,
    )

##############################################################################
