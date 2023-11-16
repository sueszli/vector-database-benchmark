from google.cloud import domains_v1

def sample_configure_contact_settings():
    if False:
        i = 10
        return i + 15
    client = domains_v1.DomainsClient()
    request = domains_v1.ConfigureContactSettingsRequest(registration='registration_value')
    operation = client.configure_contact_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)