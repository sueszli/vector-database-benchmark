from google.cloud import dialogflow_v2beta1

def sample_create_environment():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.EnvironmentsClient()
    request = dialogflow_v2beta1.CreateEnvironmentRequest(parent='parent_value', environment_id='environment_id_value')
    response = client.create_environment(request=request)
    print(response)