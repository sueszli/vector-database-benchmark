from google.cloud import securitycenter_v1

def sample_list_security_health_analytics_custom_modules():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.ListSecurityHealthAnalyticsCustomModulesRequest(parent='parent_value')
    page_result = client.list_security_health_analytics_custom_modules(request=request)
    for response in page_result:
        print(response)