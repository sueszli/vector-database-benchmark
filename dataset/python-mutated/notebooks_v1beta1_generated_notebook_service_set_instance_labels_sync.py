from google.cloud import notebooks_v1beta1

def sample_set_instance_labels():
    if False:
        i = 10
        return i + 15
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.SetInstanceLabelsRequest(name='name_value')
    operation = client.set_instance_labels(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)