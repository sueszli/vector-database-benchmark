from google.cloud import domains_v1

def sample_configure_management_settings():
    if False:
        print('Hello World!')
    client = domains_v1.DomainsClient()
    request = domains_v1.ConfigureManagementSettingsRequest(registration='registration_value')
    operation = client.configure_management_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)