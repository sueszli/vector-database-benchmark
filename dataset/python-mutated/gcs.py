import json
from pathlib import Path
from typing import Optional, Tuple
from google.cloud import storage
from google.oauth2 import service_account
from pipelines import main_logger
from pipelines.consts import GCS_PUBLIC_DOMAIN

def upload_to_gcs(file_path: Path, bucket_name: str, object_name: str, credentials: str) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    'Upload a file to a GCS bucket.\n\n    Args:\n        file_path (Path): The path to the file to upload.\n        bucket_name (str): The name of the GCS bucket.\n        object_name (str): The name of the object in the GCS bucket.\n        credentials (str): The GCS credentials as a JSON string.\n    '
    if not file_path.exists():
        main_logger.warning(f'File {file_path} does not exist. Skipping upload to GCS.')
        return ('', '')
    credentials = service_account.Credentials.from_service_account_info(json.loads(credentials))
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(file_path))
    gcs_uri = f'gs://{bucket_name}/{object_name}'
    public_url = f'{GCS_PUBLIC_DOMAIN}/{bucket_name}/{object_name}'
    return (gcs_uri, public_url)

def sanitize_gcs_credentials(raw_value: Optional[str]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Try to parse the raw string input that should contain a json object with the GCS credentials.\n    It will raise an exception if the parsing fails and help us to fail fast on invalid credentials input.\n\n    Args:\n        raw_value (str): A string representing a json object with the GCS credentials.\n\n    Returns:\n        str: The raw value string if it was successfully parsed.\n    '
    if raw_value is None:
        return None
    return json.dumps(json.loads(raw_value))