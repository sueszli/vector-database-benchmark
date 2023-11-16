from google.cloud import securitycenter_v1beta1

def sample_get_organization_settings():
    if False:
        while True:
            i = 10
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.GetOrganizationSettingsRequest(name='name_value')
    response = client.get_organization_settings(request=request)
    print(response)