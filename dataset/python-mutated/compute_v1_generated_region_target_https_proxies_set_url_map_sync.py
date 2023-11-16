from google.cloud import compute_v1

def sample_set_url_map():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionTargetHttpsProxiesClient()
    request = compute_v1.SetUrlMapRegionTargetHttpsProxyRequest(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
    response = client.set_url_map(request=request)
    print(response)