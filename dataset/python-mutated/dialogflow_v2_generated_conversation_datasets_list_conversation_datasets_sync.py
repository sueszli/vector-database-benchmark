from google.cloud import dialogflow_v2

def sample_list_conversation_datasets():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.ConversationDatasetsClient()
    request = dialogflow_v2.ListConversationDatasetsRequest(parent='parent_value')
    page_result = client.list_conversation_datasets(request=request)
    for response in page_result:
        print(response)