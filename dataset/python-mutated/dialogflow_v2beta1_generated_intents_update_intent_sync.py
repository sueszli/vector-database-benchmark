from google.cloud import dialogflow_v2beta1

def sample_update_intent():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.IntentsClient()
    intent = dialogflow_v2beta1.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflow_v2beta1.UpdateIntentRequest(intent=intent)
    response = client.update_intent(request=request)
    print(response)