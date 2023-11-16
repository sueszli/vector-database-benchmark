from google.cloud import dialogflow_v2beta1

def sample_create_context():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ContextsClient()
    request = dialogflow_v2beta1.CreateContextRequest(parent='parent_value')
    response = client.create_context(request=request)
    print(response)