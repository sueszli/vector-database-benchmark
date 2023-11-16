from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.RegionTargetHttpsProxiesClient()
    request = compute_v1.PatchRegionTargetHttpsProxyRequest(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
    response = client.patch(request=request)
    print(response)