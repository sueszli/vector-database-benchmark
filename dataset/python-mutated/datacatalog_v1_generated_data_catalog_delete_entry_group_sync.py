from google.cloud import datacatalog_v1

def sample_delete_entry_group():
    if False:
        print('Hello World!')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.DeleteEntryGroupRequest(name='name_value')
    client.delete_entry_group(request=request)