from google.cloud import datacatalog_v1

def sample_list_entries():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.ListEntriesRequest(parent='parent_value')
    page_result = client.list_entries(request=request)
    for response in page_result:
        print(response)