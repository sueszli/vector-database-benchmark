from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetHttpProxiesClient()
    request = compute_v1.DeleteTargetHttpProxyRequest(project='project_value', target_http_proxy='target_http_proxy_value')
    response = client.delete(request=request)
    print(response)