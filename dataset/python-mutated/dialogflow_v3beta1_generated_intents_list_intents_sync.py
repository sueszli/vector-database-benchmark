from google.cloud import dialogflowcx_v3beta1

def sample_list_intents():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.IntentsClient()
    request = dialogflowcx_v3beta1.ListIntentsRequest(parent='parent_value')
    page_result = client.list_intents(request=request)
    for response in page_result:
        print(response)