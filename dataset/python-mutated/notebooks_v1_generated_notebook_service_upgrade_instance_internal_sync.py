from google.cloud import notebooks_v1

def sample_upgrade_instance_internal():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.UpgradeInstanceInternalRequest(name='name_value', vm_id='vm_id_value')
    operation = client.upgrade_instance_internal(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)