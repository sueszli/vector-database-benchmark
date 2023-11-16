from google.cloud import datacatalog_v1

def sample_unstar_entry():
    if False:
        return 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.UnstarEntryRequest(name='name_value')
    response = client.unstar_entry(request=request)
    print(response)