from google.cloud import dialogflowcx_v3beta1

def sample_get_generative_settings():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.GetGenerativeSettingsRequest(name='name_value', language_code='language_code_value')
    response = client.get_generative_settings(request=request)
    print(response)