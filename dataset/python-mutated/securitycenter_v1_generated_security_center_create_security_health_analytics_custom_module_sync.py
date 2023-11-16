from google.cloud import securitycenter_v1

def sample_create_security_health_analytics_custom_module():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.CreateSecurityHealthAnalyticsCustomModuleRequest(parent='parent_value')
    response = client.create_security_health_analytics_custom_module(request=request)
    print(response)