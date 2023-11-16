from google.cloud import compute_v1

def sample_set_url_map():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetHttpProxiesClient()
    request = compute_v1.SetUrlMapTargetHttpProxyRequest(project='project_value', target_http_proxy='target_http_proxy_value')
    response = client.set_url_map(request=request)
    print(response)