from google.cloud import dialogflow_v2

def sample_list_contexts():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ContextsClient()
    request = dialogflow_v2.ListContextsRequest(parent='parent_value')
    page_result = client.list_contexts(request=request)
    for response in page_result:
        print(response)