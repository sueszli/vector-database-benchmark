from google.cloud import notebooks_v2

def sample_rollback_instance():
    if False:
        print('Hello World!')
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.RollbackInstanceRequest(name='name_value', target_snapshot='target_snapshot_value', revision_id='revision_id_value')
    operation = client.rollback_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)