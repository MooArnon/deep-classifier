##########
# Import #
##############################################################################

from datetime import datetime, timezone
import os
import traceback

from space_time_pipeline.nosql.dynamo_db import DynamoDB

from deep_classifier.utilities.logger import get_logger
from deep_classifier.main import predict

###########
# Statics #
##############################################################################

position_mapper = { 0: "SHORT", 1: "LONG"}

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
    dynamo = DynamoDB(logger)
    
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
        
        now = datetime.now(tz=timezone.utc)
        floored_time = floor_to_quarter(now)
        formatted_time = floored_time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_time_id = floored_time.strftime("%Y%m%d_%H%M%S")
        pred_id = f"{asset}_{formatted_time_id}"
        pay_load = {
            "id": pred_id,
            'asset': asset,
            "predicted_time": formatted_time,
            "model_id": model_id,
            "position": position_mapper[pred],
            "updated_at": str(now),
        }
        dynamo.ingest_data('prediction_stream', pay_load, expire_day=7)
        
        return {
            "statusCode": 200,
            "model_id": model_id,
            "position": position_mapper[pred],
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

def floor_to_quarter(dt):
    minute = (dt.minute // 15) * 15  # Floor minute to nearest quarter-hour
    return dt.replace(minute=minute, second=0, microsecond=0)

##############################################################################

if __name__ == "__main__":
    result = handler({}, None)
    print(result)
