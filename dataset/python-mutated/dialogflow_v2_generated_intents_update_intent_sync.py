from google.cloud import dialogflow_v2

def sample_update_intent():
    if False:
        return 10
    client = dialogflow_v2.IntentsClient()
    intent = dialogflow_v2.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflow_v2.UpdateIntentRequest(intent=intent)
    response = client.update_intent(request=request)
    print(response)