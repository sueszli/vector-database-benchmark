from google.cloud import alloydb_v1

def sample_list_supported_database_flags():
    if False:
        return 10
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.ListSupportedDatabaseFlagsRequest(parent='parent_value')
    page_result = client.list_supported_database_flags(request=request)
    for response in page_result:
        print(response)