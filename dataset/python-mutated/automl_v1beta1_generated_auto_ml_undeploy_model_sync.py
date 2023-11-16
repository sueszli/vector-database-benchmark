from google.cloud import automl_v1beta1

def sample_undeploy_model():
    if False:
        return 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.UndeployModelRequest(name='name_value')
    operation = client.undeploy_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)