from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetSslProxiesClient()
    request = compute_v1.DeleteTargetSslProxyRequest(project='project_value', target_ssl_proxy='target_ssl_proxy_value')
    response = client.delete(request=request)
    print(response)