from google.cloud.security import privateca_v1beta1

def sample_get_certificate():
    if False:
        print('Hello World!')
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.GetCertificateRequest(name='name_value')
    response = client.get_certificate(request=request)
    print(response)