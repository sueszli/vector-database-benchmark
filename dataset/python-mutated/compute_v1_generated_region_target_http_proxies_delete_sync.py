from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionTargetHttpProxiesClient()
    request = compute_v1.DeleteRegionTargetHttpProxyRequest(project='project_value', region='region_value', target_http_proxy='target_http_proxy_value')
    response = client.delete(request=request)
    print(response)