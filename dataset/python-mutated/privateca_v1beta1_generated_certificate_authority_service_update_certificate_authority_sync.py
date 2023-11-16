from google.cloud.security import privateca_v1beta1

def sample_update_certificate_authority():
    if False:
        return 10
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    certificate_authority = privateca_v1beta1.CertificateAuthority()
    certificate_authority.type_ = 'SUBORDINATE'
    certificate_authority.tier = 'DEVOPS'
    certificate_authority.config.reusable_config.reusable_config = 'reusable_config_value'
    certificate_authority.key_spec.cloud_kms_key_version = 'cloud_kms_key_version_value'
    request = privateca_v1beta1.UpdateCertificateAuthorityRequest(certificate_authority=certificate_authority)
    operation = client.update_certificate_authority(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)