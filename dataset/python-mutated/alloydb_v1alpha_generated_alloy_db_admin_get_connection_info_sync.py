from google.cloud import alloydb_v1alpha

def sample_get_connection_info():
    if False:
        while True:
            i = 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.GetConnectionInfoRequest(parent='parent_value')
    response = client.get_connection_info(request=request)
    print(response)