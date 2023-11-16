from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetTcpProxiesClient()
    request = compute_v1.GetTargetTcpProxyRequest(project='project_value', target_tcp_proxy='target_tcp_proxy_value')
    response = client.get(request=request)
    print(response)