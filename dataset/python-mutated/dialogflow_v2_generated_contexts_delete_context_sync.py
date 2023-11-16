from google.cloud import dialogflow_v2

def sample_delete_context():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ContextsClient()
    request = dialogflow_v2.DeleteContextRequest(name='name_value')
    client.delete_context(request=request)