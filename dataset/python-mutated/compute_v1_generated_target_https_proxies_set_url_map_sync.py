from google.cloud import compute_v1

def sample_set_url_map():
    if False:
        print('Hello World!')
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.SetUrlMapTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.set_url_map(request=request)
    print(response)