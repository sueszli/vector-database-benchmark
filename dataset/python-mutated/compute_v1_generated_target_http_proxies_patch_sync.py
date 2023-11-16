from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.TargetHttpProxiesClient()
    request = compute_v1.PatchTargetHttpProxyRequest(project='project_value', target_http_proxy='target_http_proxy_value')
    response = client.patch(request=request)
    print(response)