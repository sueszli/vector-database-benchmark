from google.cloud import alloydb_v1

def sample_restart_instance():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.RestartInstanceRequest(name='name_value')
    operation = client.restart_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)