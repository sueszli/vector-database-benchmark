from google.cloud import automl_v1

def sample_deploy_model():
    if False:
        while True:
            i = 10
    client = automl_v1.AutoMlClient()
    request = automl_v1.DeployModelRequest(name='name_value')
    operation = client.deploy_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)