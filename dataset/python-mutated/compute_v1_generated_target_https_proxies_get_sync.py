from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.GetTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.get(request=request)
    print(response)