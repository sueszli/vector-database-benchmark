from google.cloud import dialogflowcx_v3

def sample_delete_intent():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.IntentsClient()
    request = dialogflowcx_v3.DeleteIntentRequest(name='name_value')
    client.delete_intent(request=request)