from google.cloud import dialogflow_v2beta1

def sample_get_context():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ContextsClient()
    request = dialogflow_v2beta1.GetContextRequest(name='name_value')
    response = client.get_context(request=request)
    print(response)