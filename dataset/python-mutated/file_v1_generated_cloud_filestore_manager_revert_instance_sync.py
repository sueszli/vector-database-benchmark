from google.cloud import filestore_v1

def sample_revert_instance():
    if False:
        for i in range(10):
            print('nop')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.RevertInstanceRequest(name='name_value', target_snapshot_id='target_snapshot_id_value')
    operation = client.revert_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)