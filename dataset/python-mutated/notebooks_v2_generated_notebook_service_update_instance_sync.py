from google.cloud import notebooks_v2

def sample_update_instance():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v2.NotebookServiceClient()
    instance = notebooks_v2.Instance()
    instance.gce_setup.vm_image.name = 'name_value'
    instance.gce_setup.vm_image.project = 'project_value'
    request = notebooks_v2.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)