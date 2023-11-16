from google.cloud import compute_v1

def sample_set_proxy_header():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetTcpProxiesClient()
    request = compute_v1.SetProxyHeaderTargetTcpProxyRequest(project='project_value', target_tcp_proxy='target_tcp_proxy_value')
    response = client.set_proxy_header(request=request)
    print(response)