from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.RegionTargetTcpProxiesClient()
    request = compute_v1.GetRegionTargetTcpProxyRequest(project='project_value', region='region_value', target_tcp_proxy='target_tcp_proxy_value')
    response = client.get(request=request)
    print(response)