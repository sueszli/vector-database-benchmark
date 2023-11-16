from google.cloud import datacatalog_v1beta1

def sample_lookup_entry():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.LookupEntryRequest(linked_resource='linked_resource_value')
    response = client.lookup_entry(request=request)
    print(response)