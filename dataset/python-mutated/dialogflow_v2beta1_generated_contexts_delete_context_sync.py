from google.cloud import dialogflow_v2beta1

def sample_delete_context():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.ContextsClient()
    request = dialogflow_v2beta1.DeleteContextRequest(name='name_value')
    client.delete_context(request=request)