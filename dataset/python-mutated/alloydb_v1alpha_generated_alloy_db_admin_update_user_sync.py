from google.cloud import alloydb_v1alpha

def sample_update_user():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.UpdateUserRequest()
    response = client.update_user(request=request)
    print(response)