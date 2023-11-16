from google.cloud import alloydb_v1beta

def sample_get_user():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.GetUserRequest(name='name_value')
    response = client.get_user(request=request)
    print(response)