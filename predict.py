##########
# Import #
##############################################################################

import argparse
import os

import polars as pl

from deep_classifier.main import predict

###########
# Statics #
##############################################################################

#############
# Functions #
##############################################################################

def main(
        asset: str
) -> None:

    pred = predict(
        asset=asset
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
