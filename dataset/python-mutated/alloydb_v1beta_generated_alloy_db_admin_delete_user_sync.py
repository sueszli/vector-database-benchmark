from google.cloud import alloydb_v1beta

def sample_delete_user():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.DeleteUserRequest(name='name_value')
    client.delete_user(request=request)