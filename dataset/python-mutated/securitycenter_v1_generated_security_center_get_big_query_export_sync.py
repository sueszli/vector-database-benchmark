from google.cloud import securitycenter_v1

def sample_get_big_query_export():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetBigQueryExportRequest(name='name_value')
    response = client.get_big_query_export(request=request)
    print(response)