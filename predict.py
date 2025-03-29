##########
# Import #
##############################################################################

import argparse
import os

import polars as pl

from deep_classifier.utilities.logger import get_logger
from deep_classifier.main import predict

###########
# Statics #
##############################################################################

logger = get_logger(logger_name=os.path.basename(__file__))

#############
# Functions #
##############################################################################

def main(asset: str) -> None:

    pred = predict(
        asset=asset, 
        logger=logger,
    )
    
    print(pred)

##############################################################################

def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", help="Stamp column")
    
    return parser.parse_args()

###########
# Running #
##############################################################################

if __name__ == "__main__":
    
    args = setup_args()
    main(asset=args.asset)

##############################################################################
