from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionTargetTcpProxiesClient()
    request = compute_v1.DeleteRegionTargetTcpProxyRequest(project='project_value', region='region_value', target_tcp_proxy='target_tcp_proxy_value')
    response = client.delete(request=request)
    print(response)