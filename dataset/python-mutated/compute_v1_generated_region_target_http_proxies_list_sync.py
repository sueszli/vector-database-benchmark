from google.cloud import compute_v1

def sample_list():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionTargetHttpProxiesClient()
    request = compute_v1.ListRegionTargetHttpProxiesRequest(project='project_value', region='region_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)