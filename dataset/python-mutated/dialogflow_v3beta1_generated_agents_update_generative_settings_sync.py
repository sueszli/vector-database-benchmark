from google.cloud import dialogflowcx_v3beta1

def sample_update_generative_settings():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.UpdateGenerativeSettingsRequest()
    response = client.update_generative_settings(request=request)
    print(response)