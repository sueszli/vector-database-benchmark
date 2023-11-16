from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetGrpcProxiesClient()
    request = compute_v1.GetTargetGrpcProxyRequest(project='project_value', target_grpc_proxy='target_grpc_proxy_value')
    response = client.get(request=request)
    print(response)