from google.cloud import documentai_v1beta3

def sample_update_dataset():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentServiceClient()
    dataset = documentai_v1beta3.Dataset()
    dataset.state = 'INITIALIZED'
    request = documentai_v1beta3.UpdateDatasetRequest(dataset=dataset)
    operation = client.update_dataset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)