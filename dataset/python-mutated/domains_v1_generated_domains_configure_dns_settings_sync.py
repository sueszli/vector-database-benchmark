from google.cloud import domains_v1

def sample_configure_dns_settings():
    if False:
        for i in range(10):
            print('nop')
    client = domains_v1.DomainsClient()
    request = domains_v1.ConfigureDnsSettingsRequest(registration='registration_value')
    operation = client.configure_dns_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)