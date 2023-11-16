from google.cloud import compute_v1

def sample_set_proxy_header():
    if False:
        print('Hello World!')
    client = compute_v1.TargetSslProxiesClient()
    request = compute_v1.SetProxyHeaderTargetSslProxyRequest(project='project_value', target_ssl_proxy='target_ssl_proxy_value')
    response = client.set_proxy_header(request=request)
    print(response)