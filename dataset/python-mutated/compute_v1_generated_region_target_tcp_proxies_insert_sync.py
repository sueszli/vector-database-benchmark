from google.cloud import compute_v1

def sample_insert():
    if False:
        return 10
    client = compute_v1.RegionTargetTcpProxiesClient()
    request = compute_v1.InsertRegionTargetTcpProxyRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)