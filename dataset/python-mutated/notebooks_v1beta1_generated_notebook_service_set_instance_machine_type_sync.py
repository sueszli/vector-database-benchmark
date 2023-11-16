from google.cloud import notebooks_v1beta1

def sample_set_instance_machine_type():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.SetInstanceMachineTypeRequest(name='name_value', machine_type='machine_type_value')
    operation = client.set_instance_machine_type(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)