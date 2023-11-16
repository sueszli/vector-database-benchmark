from google.cloud import dialogflowcx_v3

def sample_update_intent():
    if False:
        return 10
    client = dialogflowcx_v3.IntentsClient()
    intent = dialogflowcx_v3.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateIntentRequest(intent=intent)
    response = client.update_intent(request=request)
    print(response)