from google.cloud import alloydb_v1beta

def sample_get_instance():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)