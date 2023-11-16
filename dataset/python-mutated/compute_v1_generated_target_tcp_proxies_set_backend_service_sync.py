from google.cloud import compute_v1

def sample_set_backend_service():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetTcpProxiesClient()
    request = compute_v1.SetBackendServiceTargetTcpProxyRequest(project='project_value', target_tcp_proxy='target_tcp_proxy_value')
    response = client.set_backend_service(request=request)
    print(response)