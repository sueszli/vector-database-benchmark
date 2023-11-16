from google.cloud import automl_v1

def sample_delete_dataset():
    if False:
        print('Hello World!')
    client = automl_v1.AutoMlClient()
    request = automl_v1.DeleteDatasetRequest(name='name_value')
    operation = client.delete_dataset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)