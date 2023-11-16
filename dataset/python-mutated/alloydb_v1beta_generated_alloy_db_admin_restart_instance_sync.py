from google.cloud import alloydb_v1beta

def sample_restart_instance():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.RestartInstanceRequest(name='name_value')
    operation = client.restart_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)