from google.cloud import notebooks_v1beta1

def sample_upgrade_instance_internal():
    if False:
        while True:
            i = 10
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.UpgradeInstanceInternalRequest(name='name_value', vm_id='vm_id_value')
    operation = client.upgrade_instance_internal(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)