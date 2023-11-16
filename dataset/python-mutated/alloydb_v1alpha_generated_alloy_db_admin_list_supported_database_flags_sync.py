from google.cloud import alloydb_v1alpha

def sample_list_supported_database_flags():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.ListSupportedDatabaseFlagsRequest(parent='parent_value')
    page_result = client.list_supported_database_flags(request=request)
    for response in page_result:
        print(response)