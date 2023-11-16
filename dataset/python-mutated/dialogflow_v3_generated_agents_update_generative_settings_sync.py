from google.cloud import dialogflowcx_v3

def sample_update_generative_settings():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.UpdateGenerativeSettingsRequest()
    response = client.update_generative_settings(request=request)
    print(response)