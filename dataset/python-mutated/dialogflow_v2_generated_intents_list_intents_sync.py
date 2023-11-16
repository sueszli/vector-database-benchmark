from google.cloud import dialogflow_v2

def sample_list_intents():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.IntentsClient()
    request = dialogflow_v2.ListIntentsRequest(parent='parent_value')
    page_result = client.list_intents(request=request)
    for response in page_result:
        print(response)