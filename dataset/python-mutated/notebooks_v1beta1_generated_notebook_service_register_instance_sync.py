from google.cloud import notebooks_v1beta1

def sample_register_instance():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.RegisterInstanceRequest(parent='parent_value', instance_id='instance_id_value')
    operation = client.register_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)