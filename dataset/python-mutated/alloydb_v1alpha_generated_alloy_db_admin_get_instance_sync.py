from google.cloud import alloydb_v1alpha

def sample_get_instance():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)