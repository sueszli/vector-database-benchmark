from google.cloud import compute_v1

def sample_add_signed_url_key():
    if False:
        i = 10
        return i + 15
    client = compute_v1.BackendBucketsClient()
    request = compute_v1.AddSignedUrlKeyBackendBucketRequest(backend_bucket='backend_bucket_value', project='project_value')
    response = client.add_signed_url_key(request=request)
    print(response)