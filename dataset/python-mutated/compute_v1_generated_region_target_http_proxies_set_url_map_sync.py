from google.cloud import compute_v1

def sample_set_url_map():
    if False:
        return 10
    client = compute_v1.RegionTargetHttpProxiesClient()
    request = compute_v1.SetUrlMapRegionTargetHttpProxyRequest(project='project_value', region='region_value', target_http_proxy='target_http_proxy_value')
    response = client.set_url_map(request=request)
    print(response)