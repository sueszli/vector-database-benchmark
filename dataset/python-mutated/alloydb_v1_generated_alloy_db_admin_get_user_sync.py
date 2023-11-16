from google.cloud import alloydb_v1

def sample_get_user():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.GetUserRequest(name='name_value')
    response = client.get_user(request=request)
    print(response)