from google.cloud import automl_v1beta1

def sample_delete_dataset():
    if False:
        return 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.DeleteDatasetRequest(name='name_value')
    operation = client.delete_dataset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)