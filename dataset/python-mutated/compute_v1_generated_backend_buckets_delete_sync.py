from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.BackendBucketsClient()
    request = compute_v1.DeleteBackendBucketRequest(backend_bucket='backend_bucket_value', project='project_value')
    response = client.delete(request=request)
    print(response)