from google.cloud import notebooks_v1

def sample_update_shielded_instance_config():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.UpdateShieldedInstanceConfigRequest(name='name_value')
    operation = client.update_shielded_instance_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)