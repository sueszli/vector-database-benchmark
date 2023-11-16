from google.cloud import datacatalog_v1

def sample_delete_entry():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.DeleteEntryRequest(name='name_value')
    client.delete_entry(request=request)