from google.cloud import metastore_v1beta

def sample_list_services():
    if False:
        while True:
            i = 10
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.ListServicesRequest(parent='parent_value')
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)