from google.cloud import dialogflow_v2

def sample_get_context():
    if False:
        return 10
    client = dialogflow_v2.ContextsClient()
    request = dialogflow_v2.GetContextRequest(name='name_value')
    response = client.get_context(request=request)
    print(response)