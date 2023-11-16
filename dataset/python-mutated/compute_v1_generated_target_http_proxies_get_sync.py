from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetHttpProxiesClient()
    request = compute_v1.GetTargetHttpProxyRequest(project='project_value', target_http_proxy='target_http_proxy_value')
    response = client.get(request=request)
    print(response)