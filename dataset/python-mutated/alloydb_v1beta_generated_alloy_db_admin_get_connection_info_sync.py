from google.cloud import alloydb_v1beta

def sample_get_connection_info():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.GetConnectionInfoRequest(parent='parent_value')
    response = client.get_connection_info(request=request)
    print(response)