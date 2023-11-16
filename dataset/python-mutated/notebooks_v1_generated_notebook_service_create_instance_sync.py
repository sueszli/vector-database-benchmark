from google.cloud import notebooks_v1

def sample_create_instance():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1.NotebookServiceClient()
    instance = notebooks_v1.Instance()
    instance.vm_image.image_name = 'image_name_value'
    instance.vm_image.project = 'project_value'
    instance.machine_type = 'machine_type_value'
    request = notebooks_v1.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)