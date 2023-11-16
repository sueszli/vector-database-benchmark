from google.cloud import alloydb_v1beta

def sample_update_user():
    if False:
        print('Hello World!')
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.UpdateUserRequest()
    response = client.update_user(request=request)
    print(response)