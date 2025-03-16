##########
# Import #
##############################################################################

import os
import traceback

from deep_classifier.utilities.logger import get_logger
from deep_classifier.main import predict

#################
# Lambda handel #
##############################################################################

def handler(event = None, context = None):
    """
    AWS Lambda handler function.
    
    event: a dict, must contain an 'assets' key with a list of asset symbols.
    context: Lambda context object (not used here).
    """
    logger = get_logger(logger_name=os.path.basename(__file__))
    
    logger.info("Handler function started.")
    file_path = None
    
    # Get the list of assets from the event input
    asset = os.environ['ASSET']

    # Call the main function with the assets list
    try:
        pred, model_id = predict(
            asset=asset, 
            logger=logger,
        )
        
        logger.info(f"Predicted result: {pred} from {model_id}")
        return {
            "statusCode": 200,
            "model_id": model_id,
            "prediction": pred,
            "body": "Prediction completed successfully."
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file {file_path} removed.")

##############################################################################

if __name__ == "__main__":
    handler({}, None)
