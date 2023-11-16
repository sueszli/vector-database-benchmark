from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionTargetHttpProxiesClient()
    request = compute_v1.GetRegionTargetHttpProxyRequest(project='project_value', region='region_value', target_http_proxy='target_http_proxy_value')
    response = client.get(request=request)
    print(response)