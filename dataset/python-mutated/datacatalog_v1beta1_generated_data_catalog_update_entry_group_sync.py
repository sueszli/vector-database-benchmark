from google.cloud import datacatalog_v1beta1

def sample_update_entry_group():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.UpdateEntryGroupRequest()
    response = client.update_entry_group(request=request)
    print(response)