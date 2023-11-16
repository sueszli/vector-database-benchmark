from google.cloud import automl_v1

def sample_undeploy_model():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1.AutoMlClient()
    request = automl_v1.UndeployModelRequest(name='name_value')
    operation = client.undeploy_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)