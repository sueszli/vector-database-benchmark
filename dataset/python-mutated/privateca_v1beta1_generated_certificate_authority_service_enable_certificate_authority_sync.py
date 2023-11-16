from google.cloud.security import privateca_v1beta1

def sample_enable_certificate_authority():
    if False:
        i = 10
        return i + 15
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.EnableCertificateAuthorityRequest(name='name_value')
    operation = client.enable_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)