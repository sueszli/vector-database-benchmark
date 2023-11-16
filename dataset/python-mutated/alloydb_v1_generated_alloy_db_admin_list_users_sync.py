from google.cloud import alloydb_v1

def sample_list_users():
    if False:
        while True:
            i = 10
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.ListUsersRequest(parent='parent_value')
    page_result = client.list_users(request=request)
    for response in page_result:
        print(response)