from google.cloud import dialogflowcx_v3beta1

def sample_create_environment():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    environment = dialogflowcx_v3beta1.Environment()
    environment.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.CreateEnvironmentRequest(parent='parent_value', environment=environment)
    operation = client.create_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)