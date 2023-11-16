from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionTargetHttpsProxiesClient()
    request = compute_v1.GetRegionTargetHttpsProxyRequest(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
    response = client.get(request=request)
    print(response)