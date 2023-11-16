from google.cloud import datacatalog_v1beta1

def sample_get_entry():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.GetEntryRequest(name='name_value')
    response = client.get_entry(request=request)
    print(response)