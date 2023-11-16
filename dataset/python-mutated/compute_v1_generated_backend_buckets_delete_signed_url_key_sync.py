from google.cloud import compute_v1

def sample_delete_signed_url_key():
    if False:
        i = 10
        return i + 15
    client = compute_v1.BackendBucketsClient()
    request = compute_v1.DeleteSignedUrlKeyBackendBucketRequest(backend_bucket='backend_bucket_value', key_name='key_name_value', project='project_value')
    response = client.delete_signed_url_key(request=request)
    print(response)