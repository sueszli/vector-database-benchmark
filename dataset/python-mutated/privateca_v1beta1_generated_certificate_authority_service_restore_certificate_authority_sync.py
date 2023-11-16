from google.cloud.security import privateca_v1beta1

def sample_restore_certificate_authority():
    if False:
        print('Hello World!')
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.RestoreCertificateAuthorityRequest(name='name_value')
    operation = client.restore_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)