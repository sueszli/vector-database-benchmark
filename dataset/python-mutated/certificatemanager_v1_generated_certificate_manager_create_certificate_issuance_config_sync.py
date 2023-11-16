from google.cloud import certificate_manager_v1

def sample_create_certificate_issuance_config():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    certificate_issuance_config = certificate_manager_v1.CertificateIssuanceConfig()
    certificate_issuance_config.certificate_authority_config.certificate_authority_service_config.ca_pool = 'ca_pool_value'
    certificate_issuance_config.rotation_window_percentage = 2788
    certificate_issuance_config.key_algorithm = 'ECDSA_P256'
    request = certificate_manager_v1.CreateCertificateIssuanceConfigRequest(parent='parent_value', certificate_issuance_config_id='certificate_issuance_config_id_value', certificate_issuance_config=certificate_issuance_config)
    operation = client.create_certificate_issuance_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)