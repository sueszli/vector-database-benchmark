from google.cloud import securitycenter_v1

def sample_list_descendant_security_health_analytics_custom_modules():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.ListDescendantSecurityHealthAnalyticsCustomModulesRequest(parent='parent_value')
    page_result = client.list_descendant_security_health_analytics_custom_modules(request=request)
    for response in page_result:
        print(response)