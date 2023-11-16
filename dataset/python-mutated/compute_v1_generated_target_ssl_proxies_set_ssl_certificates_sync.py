from google.cloud import compute_v1

def sample_set_ssl_certificates():
    if False:
        return 10
    client = compute_v1.TargetSslProxiesClient()
    request = compute_v1.SetSslCertificatesTargetSslProxyRequest(project='project_value', target_ssl_proxy='target_ssl_proxy_value')
    response = client.set_ssl_certificates(request=request)
    print(response)