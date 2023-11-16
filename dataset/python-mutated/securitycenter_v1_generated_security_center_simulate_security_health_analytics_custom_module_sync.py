from google.cloud import securitycenter_v1

def sample_simulate_security_health_analytics_custom_module():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    resource = securitycenter_v1.SimulatedResource()
    resource.resource_type = 'resource_type_value'
    request = securitycenter_v1.SimulateSecurityHealthAnalyticsCustomModuleRequest(parent='parent_value', resource=resource)
    response = client.simulate_security_health_analytics_custom_module(request=request)
    print(response)