from google.cloud import dialogflow_v2beta1

def sample_update_context():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ContextsClient()
    request = dialogflow_v2beta1.UpdateContextRequest()
    response = client.update_context(request=request)
    print(response)