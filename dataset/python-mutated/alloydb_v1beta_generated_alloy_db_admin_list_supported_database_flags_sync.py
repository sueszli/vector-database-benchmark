from google.cloud import alloydb_v1beta

def sample_list_supported_database_flags():
    if False:
        return 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.ListSupportedDatabaseFlagsRequest(parent='parent_value')
    page_result = client.list_supported_database_flags(request=request)
    for response in page_result:
        print(response)