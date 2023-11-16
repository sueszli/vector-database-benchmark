from google.cloud import notebooks_v1

def sample_rollback_instance():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.RollbackInstanceRequest(name='name_value', target_snapshot='target_snapshot_value')
    operation = client.rollback_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)