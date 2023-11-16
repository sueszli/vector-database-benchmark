from google.cloud import compute_v1

def sample_set_ssl_certificates():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetHttpsProxiesClient()
    request = compute_v1.SetSslCertificatesTargetHttpsProxyRequest(project='project_value', target_https_proxy='target_https_proxy_value')
    response = client.set_ssl_certificates(request=request)
    print(response)