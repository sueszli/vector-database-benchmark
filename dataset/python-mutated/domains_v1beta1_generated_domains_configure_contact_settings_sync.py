from google.cloud import domains_v1beta1

def sample_configure_contact_settings():
    if False:
        for i in range(10):
            print('nop')
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.ConfigureContactSettingsRequest(registration='registration_value')
    operation = client.configure_contact_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)