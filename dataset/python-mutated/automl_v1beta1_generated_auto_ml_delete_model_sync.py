from google.cloud import automl_v1beta1

def sample_delete_model():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.DeleteModelRequest(name='name_value')
    operation = client.delete_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)