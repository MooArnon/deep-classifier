##########
# Import #
##############################################################################

import boto3
from botocore.exceptions import NoCredentialsError

##############################################################################

def clear_s3_prefix(bucket, prefix):
    """Delete all files under a specific prefix in an S3 bucket, pass if no files found."""
    s3 = boto3.client('s3')
    
    try:
        # List all the objects under the given prefix
        objects_to_delete = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' in objects_to_delete:
            # Create a list of object identifiers (Key, VersionId) for deletion
            delete_keys = [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]

            # Perform the delete operation
            s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_keys})
            print(f"All files under prefix {prefix} have been deleted from {bucket}")
        else:
            print(f"No files found under prefix {prefix} in bucket {bucket}. Continuing with upload.")
            pass  # Just pass if no files are found
        return True

    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
##############################################################################

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket.

    Parameter
    ---------
    file_name: str
        File to upload
    bucket: str
        Bucket to upload to
    object_name: str
        S3 object name. If not specified then file_name is used
    
    Return
    ------
    bool
        True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3')

    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded successfully to {bucket}/{object_name}")
        return True
    except FileNotFoundError:
        print(f"The file {file_name} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

##############################################################################

def clear_and_push_to_s3(file_name, bucket, prefix):
    """Clear a specific prefix and then upload a file to S3.

    Parameter
    ---------
    file_name: 
        The local file to upload
    bucket: 
        The name of the S3 bucket
    prefix: 
        The prefix (folder path) to clear and then upload the file to
    """
    # Step 1: Clear the target prefix
    cleared = clear_s3_prefix(bucket, prefix)
    
    # Step 2: Upload the new file to the cleared prefix
    if cleared:
        object_name = prefix + file_name.split('/')[-1]  
        upload_to_s3(file_name, bucket, object_name)

##############################################################################
