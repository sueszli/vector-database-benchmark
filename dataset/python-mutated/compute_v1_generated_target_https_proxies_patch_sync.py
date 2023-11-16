from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.PatchTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.patch(request=request)
    print(response)