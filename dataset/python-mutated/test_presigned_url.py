"""
Unit tests for presigned_url.py functions.
"""
import boto3
import presigned_url

def test_generate_presigned_url():
    if False:
        while True:
            i = 10
    s3_client = boto3.client('s3')
    client_method = 'get_object'
    method_params = {'Bucket': 'test-bucket', 'Key': 'test-key'}
    expires = 10
    got_url = presigned_url.generate_presigned_url(s3_client, client_method, method_params, expires)
    assert 'test-bucket' in got_url
    assert 'test-key' in got_url