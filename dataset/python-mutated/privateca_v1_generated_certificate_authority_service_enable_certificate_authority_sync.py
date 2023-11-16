from google.cloud.security import privateca_v1

def sample_enable_certificate_authority():
    if False:
        return 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.EnableCertificateAuthorityRequest(name='name_value')
    operation = client.enable_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)