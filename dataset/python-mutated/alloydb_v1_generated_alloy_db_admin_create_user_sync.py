from google.cloud import alloydb_v1

def sample_create_user():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.CreateUserRequest(parent='parent_value', user_id='user_id_value')
    response = client.create_user(request=request)
    print(response)