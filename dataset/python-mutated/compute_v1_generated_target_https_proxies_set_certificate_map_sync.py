from google.cloud import compute_v1

def sample_set_certificate_map():
    if False:
        return 10
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.SetCertificateMapTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.set_certificate_map(request=request)
    print(response)