from google.cloud import dialogflowcx_v3beta1

def sample_update_intent():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.IntentsClient()
    intent = dialogflowcx_v3beta1.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateIntentRequest(intent=intent)
    response = client.update_intent(request=request)
    print(response)