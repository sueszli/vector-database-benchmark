from google.cloud import securitycenter_v1

def sample_get_effective_security_health_analytics_custom_module():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetEffectiveSecurityHealthAnalyticsCustomModuleRequest(name='name_value')
    response = client.get_effective_security_health_analytics_custom_module(request=request)
    print(response)