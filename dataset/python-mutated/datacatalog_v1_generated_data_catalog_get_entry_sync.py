from google.cloud import datacatalog_v1

def sample_get_entry():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.GetEntryRequest(name='name_value')
    response = client.get_entry(request=request)
    print(response)