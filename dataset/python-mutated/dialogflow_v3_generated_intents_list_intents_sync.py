from google.cloud import dialogflowcx_v3

def sample_list_intents():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.IntentsClient()
    request = dialogflowcx_v3.ListIntentsRequest(parent='parent_value')
    page_result = client.list_intents(request=request)
    for response in page_result:
        print(response)