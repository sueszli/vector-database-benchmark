from google.cloud import alloydb_v1alpha

def sample_list_users():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.ListUsersRequest(parent='parent_value')
    page_result = client.list_users(request=request)
    for response in page_result:
        print(response)