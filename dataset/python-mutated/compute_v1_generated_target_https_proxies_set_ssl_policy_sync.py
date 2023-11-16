from google.cloud import compute_v1

def sample_set_ssl_policy():
    if False:
        print('Hello World!')
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.SetSslPolicyTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.set_ssl_policy(request=request)
    print(response)