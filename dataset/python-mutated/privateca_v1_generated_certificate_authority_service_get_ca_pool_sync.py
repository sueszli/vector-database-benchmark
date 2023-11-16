from google.cloud.security import privateca_v1

def sample_get_ca_pool():
    if False:
        return 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.GetCaPoolRequest(name='name_value')
    response = client.get_ca_pool(request=request)
    print(response)