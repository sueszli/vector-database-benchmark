from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.DeleteTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.delete(request=request)
    print(response)