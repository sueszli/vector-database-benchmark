from google.cloud import automl_v1

def sample_create_model():
    if False:
        print('Hello World!')
    client = automl_v1.AutoMlClient()
    request = automl_v1.CreateModelRequest(parent='parent_value')
    operation = client.create_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)