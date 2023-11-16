from google.cloud import dialogflow_v2beta1

def sample_list_intents():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.IntentsClient()
    request = dialogflow_v2beta1.ListIntentsRequest(parent='parent_value')
    page_result = client.list_intents(request=request)
    for response in page_result:
        print(response)