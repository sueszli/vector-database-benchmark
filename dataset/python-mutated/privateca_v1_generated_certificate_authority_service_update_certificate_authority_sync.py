from google.cloud.security import privateca_v1

def sample_update_certificate_authority():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    certificate_authority = privateca_v1.CertificateAuthority()
    certificate_authority.type_ = 'SUBORDINATE'
    certificate_authority.key_spec.cloud_kms_key_version = 'cloud_kms_key_version_value'
    request = privateca_v1.UpdateCertificateAuthorityRequest(certificate_authority=certificate_authority)
    operation = client.update_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)