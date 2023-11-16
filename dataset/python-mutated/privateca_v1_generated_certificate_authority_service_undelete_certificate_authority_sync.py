from google.cloud.security import privateca_v1

def sample_undelete_certificate_authority():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.UndeleteCertificateAuthorityRequest(name='name_value')
    operation = client.undelete_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)