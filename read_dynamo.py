from space_time_pipeline.nosql.dynamo_db import DynamoDB
from deep_classifier.utilities.logger import get_logger

logger = get_logger("read-dynamo")
dynamo = DynamoDB(logger=logger)

print(
    dynamo.print_all_records(
        table='prediction_stream'
    )
)
