from google.cloud import securitycenter_v1beta1

def sample_update_organization_settings():
    if False:
        print('Hello World!')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.UpdateOrganizationSettingsRequest()
    response = client.update_organization_settings(request=request)
    print(response)