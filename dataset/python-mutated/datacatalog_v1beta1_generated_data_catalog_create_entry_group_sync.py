from google.cloud import datacatalog_v1beta1

def sample_create_entry_group():
    if False:
        return 10
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.CreateEntryGroupRequest(parent='parent_value', entry_group_id='entry_group_id_value')
    response = client.create_entry_group(request=request)
    print(response)