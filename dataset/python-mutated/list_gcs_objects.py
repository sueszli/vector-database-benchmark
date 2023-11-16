import argparse
from typing import List
import boto3

def list_gcs_objects(google_access_key_id: str, google_access_key_secret: str, bucket_name: str) -> List[str]:
    if False:
        while True:
            i = 10
    'Lists all Cloud Storage objects using AWS SDK for Python (boto3)\n    Positional arguments:\n        google_access_key_id: hash-based message authentication code (HMAC) access ID\n        google_access_key_secret: HMAC access secret\n        bucket_name: name of Cloud Storage bucket\n\n    Returned value is a list of strings, one for each object in the bucket.\n\n    To use this sample:\n    1. Create a Cloud Storage HMAC key: https://cloud.google.com/storage/docs/authentication/managing-hmackeys#create\n    2. Change endpoint_url to a Google Cloud Storage XML API endpoint.\n\n    To learn more about HMAC: https://cloud.google.com/storage/docs/authentication/hmackeys#overview\n    '
    client = boto3.client('s3', region_name='auto', endpoint_url='https://storage.googleapis.com', aws_access_key_id=google_access_key_id, aws_secret_access_key=google_access_key_secret)
    response = client.list_objects(Bucket=bucket_name)
    results = []
    for blob in response['Contents']:
        results.append(blob['Key'])
        print(blob['Key'])
    return results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('google_access_key_id', help='Your Cloud Storage HMAC Access Key ID.')
    parser.add_argument('google_access_key_secret', help='Your Cloud Storage HMAC Access Key Secret.')
    parser.add_argument('bucket_name', help='Your Cloud Storage bucket name')
    args = parser.parse_args()
    list_gcs_objects(google_access_key_id=args.google_access_key_id, google_access_key_secret=args.google_access_key_secret, bucket_name=args.bucket_name)