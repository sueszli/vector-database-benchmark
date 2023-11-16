from google.cloud import securitycenter_v1

def sample_get_security_health_analytics_custom_module():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetSecurityHealthAnalyticsCustomModuleRequest(name='name_value')
    response = client.get_security_health_analytics_custom_module(request=request)
    print(response)