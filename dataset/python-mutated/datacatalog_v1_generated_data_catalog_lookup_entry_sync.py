from google.cloud import datacatalog_v1

def sample_lookup_entry():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.LookupEntryRequest(linked_resource='linked_resource_value')
    response = client.lookup_entry(request=request)
    print(response)