from google.cloud import dialogflow_v2beta1

def sample_list_contexts():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ContextsClient()
    request = dialogflow_v2beta1.ListContextsRequest(parent='parent_value')
    page_result = client.list_contexts(request=request)
    for response in page_result:
        print(response)