from google.cloud import metastore_v1

def sample_list_services():
    if False:
        i = 10
        return i + 15
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.ListServicesRequest(parent='parent_value')
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)