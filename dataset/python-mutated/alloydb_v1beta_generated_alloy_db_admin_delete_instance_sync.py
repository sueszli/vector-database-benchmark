from google.cloud import alloydb_v1beta

def sample_delete_instance():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)