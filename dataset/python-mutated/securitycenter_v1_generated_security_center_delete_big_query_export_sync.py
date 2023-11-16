from google.cloud import securitycenter_v1

def sample_delete_big_query_export():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.DeleteBigQueryExportRequest(name='name_value')
    client.delete_big_query_export(request=request)