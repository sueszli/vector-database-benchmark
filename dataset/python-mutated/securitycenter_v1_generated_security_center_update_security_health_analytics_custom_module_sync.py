from google.cloud import securitycenter_v1

def sample_update_security_health_analytics_custom_module():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.UpdateSecurityHealthAnalyticsCustomModuleRequest()
    response = client.update_security_health_analytics_custom_module(request=request)
    print(response)