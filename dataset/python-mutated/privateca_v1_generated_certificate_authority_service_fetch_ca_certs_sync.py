from google.cloud.security import privateca_v1

def sample_fetch_ca_certs():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.FetchCaCertsRequest(ca_pool='ca_pool_value')
    response = client.fetch_ca_certs(request=request)
    print(response)