from google.cloud import domains_v1beta1

def sample_configure_management_settings():
    if False:
        return 10
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.ConfigureManagementSettingsRequest(registration='registration_value')
    operation = client.configure_management_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)