from google.cloud import dialogflowcx_v3

def sample_update_environment():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.EnvironmentsClient()
    environment = dialogflowcx_v3.Environment()
    environment.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateEnvironmentRequest(environment=environment)
    operation = client.update_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)