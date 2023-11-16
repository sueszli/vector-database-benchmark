from google.cloud import alloydb_v1alpha

def sample_delete_user():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.DeleteUserRequest(name='name_value')
    client.delete_user(request=request)