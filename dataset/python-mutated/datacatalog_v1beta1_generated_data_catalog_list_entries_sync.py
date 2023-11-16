from google.cloud import datacatalog_v1beta1

def sample_list_entries():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.ListEntriesRequest(parent='parent_value')
    page_result = client.list_entries(request=request)
    for response in page_result:
        print(response)