from google.cloud import compute_v1

def sample_set_certificate_map():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.TargetSslProxiesClient()
    request = compute_v1.SetCertificateMapTargetSslProxyRequest(project='project_value', target_ssl_proxy='target_ssl_proxy_value')
    response = client.set_certificate_map(request=request)
    print(response)