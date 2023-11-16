from google.cloud.security import privateca_v1beta1

def sample_revoke_certificate():
    if False:
        while True:
            i = 10
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.RevokeCertificateRequest(name='name_value', reason='ATTRIBUTE_AUTHORITY_COMPROMISE')
    response = client.revoke_certificate(request=request)
    print(response)