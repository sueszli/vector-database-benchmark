from google.cloud import dialogflowcx_v3beta1

def sample_update_environment():
    if False:
        return 10
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    environment = dialogflowcx_v3beta1.Environment()
    environment.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateEnvironmentRequest(environment=environment)
    operation = client.update_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)