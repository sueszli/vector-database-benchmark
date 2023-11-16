from google.cloud import alloydb_v1

def sample_update_user():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.UpdateUserRequest()
    response = client.update_user(request=request)
    print(response)