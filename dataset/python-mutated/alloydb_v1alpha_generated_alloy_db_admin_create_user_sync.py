from google.cloud import alloydb_v1alpha

def sample_create_user():
    if False:
        return 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.CreateUserRequest(parent='parent_value', user_id='user_id_value')
    response = client.create_user(request=request)
    print(response)