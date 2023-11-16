from google.cloud.security import privateca_v1

def sample_disable_certificate_authority():
    if False:
        while True:
            i = 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.DisableCertificateAuthorityRequest(name='name_value')
    operation = client.disable_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)