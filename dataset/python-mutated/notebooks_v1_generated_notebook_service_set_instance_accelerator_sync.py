from google.cloud import notebooks_v1

def sample_set_instance_accelerator():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.SetInstanceAcceleratorRequest(name='name_value', type_='TPU_V3', core_count=1073)
    operation = client.set_instance_accelerator(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)