from google.cloud import datacatalog_v1beta1

def sample_delete_entry_group():
    if False:
        return 10
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.DeleteEntryGroupRequest(name='name_value')
    client.delete_entry_group(request=request)