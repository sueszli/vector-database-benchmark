from google.cloud.security import privateca_v1beta1

def sample_get_certificate_revocation_list():
    if False:
        print('Hello World!')
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.GetCertificateRevocationListRequest(name='name_value')
    response = client.get_certificate_revocation_list(request=request)
    print(response)