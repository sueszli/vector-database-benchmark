from google.cloud import dialogflowcx_v3

def sample_get_generative_settings():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.GetGenerativeSettingsRequest(name='name_value', language_code='language_code_value')
    response = client.get_generative_settings(request=request)
    print(response)