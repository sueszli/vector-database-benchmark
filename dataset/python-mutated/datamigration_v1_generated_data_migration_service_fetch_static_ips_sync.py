from google.cloud import clouddms_v1

def sample_fetch_static_ips():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.FetchStaticIpsRequest(name='name_value')
    page_result = client.fetch_static_ips(request=request)
    for response in page_result:
        print(response)