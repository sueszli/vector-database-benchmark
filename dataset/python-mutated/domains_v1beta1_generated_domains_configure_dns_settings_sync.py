from google.cloud import domains_v1beta1

def sample_configure_dns_settings():
    if False:
        i = 10
        return i + 15
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.ConfigureDnsSettingsRequest(registration='registration_value')
    operation = client.configure_dns_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)