from google.cloud import compute_v1

def sample_set_edge_security_policy():
    if False:
        i = 10
        return i + 15
    client = compute_v1.BackendBucketsClient()
    request = compute_v1.SetEdgeSecurityPolicyBackendBucketRequest(backend_bucket='backend_bucket_value', project='project_value')
    response = client.set_edge_security_policy(request=request)
    print(response)