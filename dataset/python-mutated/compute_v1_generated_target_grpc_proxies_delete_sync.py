from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.TargetGrpcProxiesClient()
    request = compute_v1.DeleteTargetGrpcProxyRequest(project='project_value', target_grpc_proxy='target_grpc_proxy_value')
    response = client.delete(request=request)
    print(response)