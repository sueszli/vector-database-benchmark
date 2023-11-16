from google.cloud import dialogflow_v2beta1

def sample_delete_intent():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.IntentsClient()
    request = dialogflow_v2beta1.DeleteIntentRequest(name='name_value')
    client.delete_intent(request=request)