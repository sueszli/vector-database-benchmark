from google.cloud import dialogflowcx_v3

def sample_create_environment():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.EnvironmentsClient()
    environment = dialogflowcx_v3.Environment()
    environment.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreateEnvironmentRequest(parent='parent_value', environment=environment)
    operation = client.create_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)