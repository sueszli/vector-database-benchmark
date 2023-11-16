from google.cloud import dialogflow_v2

def sample_create_context():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ContextsClient()
    context = dialogflow_v2.Context()
    context.name = 'name_value'
    request = dialogflow_v2.CreateContextRequest(parent='parent_value', context=context)
    response = client.create_context(request=request)
    print(response)