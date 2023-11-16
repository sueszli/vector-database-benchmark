from google.cloud import securitycenter_v1

def sample_get_organization_settings():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetOrganizationSettingsRequest(name='name_value')
    response = client.get_organization_settings(request=request)
    print(response)