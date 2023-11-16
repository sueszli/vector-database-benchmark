from google.cloud import datacatalog_v1

def sample_get_entry_group():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.GetEntryGroupRequest(name='name_value')
    response = client.get_entry_group(request=request)
    print(response)