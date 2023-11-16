from google.cloud import datacatalog_v1

def sample_star_entry():
    if False:
        return 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.StarEntryRequest(name='name_value')
    response = client.star_entry(request=request)
    print(response)