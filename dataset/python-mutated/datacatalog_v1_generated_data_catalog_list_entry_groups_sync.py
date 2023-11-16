from google.cloud import datacatalog_v1

def sample_list_entry_groups():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.ListEntryGroupsRequest(parent='parent_value')
    page_result = client.list_entry_groups(request=request)
    for response in page_result:
        print(response)