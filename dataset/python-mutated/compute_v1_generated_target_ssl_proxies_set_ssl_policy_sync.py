from google.cloud import compute_v1

def sample_set_ssl_policy():
    if False:
        print('Hello World!')
    client = compute_v1.TargetSslProxiesClient()
    request = compute_v1.SetSslPolicyTargetSslProxyRequest(project='project_value', target_ssl_proxy='target_ssl_proxy_value')
    response = client.set_ssl_policy(request=request)
    print(response)