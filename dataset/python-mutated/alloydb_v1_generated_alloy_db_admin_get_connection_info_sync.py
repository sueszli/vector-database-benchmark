from google.cloud import alloydb_v1

def sample_get_connection_info():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.GetConnectionInfoRequest(parent='parent_value')
    response = client.get_connection_info(request=request)
    print(response)