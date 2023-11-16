from google.cloud import notebooks_v1beta1

def sample_create_environment():
    if False:
        while True:
            i = 10
    client = notebooks_v1beta1.NotebookServiceClient()
    environment = notebooks_v1beta1.Environment()
    environment.vm_image.image_name = 'image_name_value'
    environment.vm_image.project = 'project_value'
    request = notebooks_v1beta1.CreateEnvironmentRequest(parent='parent_value', environment_id='environment_id_value', environment=environment)
    operation = client.create_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)