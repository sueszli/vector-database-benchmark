from google.cloud import compute_v1

def sample_set_ssl_certificates():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionTargetHttpsProxiesClient()
    request = compute_v1.SetSslCertificatesRegionTargetHttpsProxyRequest(project='project_value', region='region_value', target_https_proxy='target_https_proxy_value')
    response = client.set_ssl_certificates(request=request)
    print(response)