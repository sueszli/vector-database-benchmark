from google.cloud import dialogflow_v2

def sample_delete_all_contexts():
    if False:
        return 10
    client = dialogflow_v2.ContextsClient()
    request = dialogflow_v2.DeleteAllContextsRequest(parent='parent_value')
    client.delete_all_contexts(request=request)