from google.cloud import dialogflow_v2

def sample_delete_intent():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.IntentsClient()
    request = dialogflow_v2.DeleteIntentRequest(name='name_value')
    client.delete_intent(request=request)