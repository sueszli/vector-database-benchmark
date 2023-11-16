from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.BackendBucketsClient()
    request = compute_v1.PatchBackendBucketRequest(backend_bucket='backend_bucket_value', project='project_value')
    response = client.patch(request=request)
    print(response)