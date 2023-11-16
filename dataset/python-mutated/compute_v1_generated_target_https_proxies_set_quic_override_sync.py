from google.cloud import compute_v1

def sample_set_quic_override():
    if False:
        i = 10
        return i + 15
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.SetQuicOverrideTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.set_quic_override(request=request)
    print(response)