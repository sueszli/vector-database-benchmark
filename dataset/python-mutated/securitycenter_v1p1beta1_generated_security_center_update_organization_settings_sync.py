from google.cloud import securitycenter_v1p1beta1

def sample_update_organization_settings():
    if False:
        return 10
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.UpdateOrganizationSettingsRequest()
    response = client.update_organization_settings(request=request)
    print(response)