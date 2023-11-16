from google.cloud import securitycenter_v1

def sample_create_big_query_export():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.CreateBigQueryExportRequest(parent='parent_value', big_query_export_id='big_query_export_id_value')
    response = client.create_big_query_export(request=request)
    print(response)