from google.cloud import datacatalog_v1

def sample_update_entry_group():
    if False:
        return 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.UpdateEntryGroupRequest()
    response = client.update_entry_group(request=request)
    print(response)