from google.cloud import notebooks_v2

def sample_create_instance():
    if False:
        print('Hello World!')
    client = notebooks_v2.NotebookServiceClient()
    instance = notebooks_v2.Instance()
    instance.gce_setup.vm_image.name = 'name_value'
    instance.gce_setup.vm_image.project = 'project_value'
    request = notebooks_v2.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)