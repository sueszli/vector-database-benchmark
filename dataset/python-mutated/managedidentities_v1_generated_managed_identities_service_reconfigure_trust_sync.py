from google.cloud import managedidentities_v1

def sample_reconfigure_trust():
    if False:
        i = 10
        return i + 15
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    request = managedidentities_v1.ReconfigureTrustRequest(name='name_value', target_domain_name='target_domain_name_value', target_dns_ip_addresses=['target_dns_ip_addresses_value1', 'target_dns_ip_addresses_value2'])
    operation = client.reconfigure_trust(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)