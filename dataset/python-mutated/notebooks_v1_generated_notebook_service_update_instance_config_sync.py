from google.cloud import notebooks_v1

def sample_update_instance_config():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.UpdateInstanceConfigRequest(name='name_value')
    operation = client.update_instance_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)