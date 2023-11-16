from google.cloud import dialogflow_v2

def sample_update_context():
    if False:
        return 10
    client = dialogflow_v2.ContextsClient()
    context = dialogflow_v2.Context()
    context.name = 'name_value'
    request = dialogflow_v2.UpdateContextRequest(context=context)
    response = client.update_context(request=request)
    print(response)