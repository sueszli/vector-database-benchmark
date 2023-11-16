from google.cloud import alloydb_v1

def sample_delete_user():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.DeleteUserRequest(name='name_value')
    client.delete_user(request=request)