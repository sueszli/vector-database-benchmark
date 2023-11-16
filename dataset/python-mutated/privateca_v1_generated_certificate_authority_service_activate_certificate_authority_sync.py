from google.cloud.security import privateca_v1

def sample_activate_certificate_authority():
    if False:
        return 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    subordinate_config = privateca_v1.SubordinateConfig()
    subordinate_config.certificate_authority = 'certificate_authority_value'
    request = privateca_v1.ActivateCertificateAuthorityRequest(name='name_value', pem_ca_certificate='pem_ca_certificate_value', subordinate_config=subordinate_config)
    operation = client.activate_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)