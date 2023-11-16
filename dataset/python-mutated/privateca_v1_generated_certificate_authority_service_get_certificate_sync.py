from google.cloud.security import privateca_v1

def sample_get_certificate():
    if False:
        return 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.GetCertificateRequest(name='name_value')
    response = client.get_certificate(request=request)
    print(response)