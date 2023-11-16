from google.cloud import dialogflow_v2beta1

def sample_list_environments():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.EnvironmentsClient()
    request = dialogflow_v2beta1.ListEnvironmentsRequest(parent='parent_value')
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)