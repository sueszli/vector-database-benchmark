from google.cloud import securitycenter_v1

def sample_delete_security_health_analytics_custom_module():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.DeleteSecurityHealthAnalyticsCustomModuleRequest(name='name_value')
    client.delete_security_health_analytics_custom_module(request=request)