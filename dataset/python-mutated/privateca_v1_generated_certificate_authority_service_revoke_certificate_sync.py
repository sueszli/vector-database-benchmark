from google.cloud.security import privateca_v1

def sample_revoke_certificate():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.RevokeCertificateRequest(name='name_value', reason='ATTRIBUTE_AUTHORITY_COMPROMISE')
    response = client.revoke_certificate(request=request)
    print(response)