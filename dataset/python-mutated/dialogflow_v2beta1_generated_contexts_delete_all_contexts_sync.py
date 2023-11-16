from google.cloud import dialogflow_v2beta1

def sample_delete_all_contexts():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.ContextsClient()
    request = dialogflow_v2beta1.DeleteAllContextsRequest(parent='parent_value')
    client.delete_all_contexts(request=request)