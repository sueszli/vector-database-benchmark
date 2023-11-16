from google.cloud import managedidentities_v1

def sample_validate_trust():
    if False:
        return 10
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    trust = managedidentities_v1.Trust()
    trust.target_domain_name = 'target_domain_name_value'
    trust.trust_type = 'EXTERNAL'
    trust.trust_direction = 'BIDIRECTIONAL'
    trust.target_dns_ip_addresses = ['target_dns_ip_addresses_value1', 'target_dns_ip_addresses_value2']
    trust.trust_handshake_secret = 'trust_handshake_secret_value'
    request = managedidentities_v1.ValidateTrustRequest(name='name_value', trust=trust)
    operation = client.validate_trust(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)