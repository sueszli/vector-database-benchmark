from google.cloud import datacatalog_v1beta1

def sample_get_entry_group():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.GetEntryGroupRequest(name='name_value')
    response = client.get_entry_group(request=request)
    print(response)