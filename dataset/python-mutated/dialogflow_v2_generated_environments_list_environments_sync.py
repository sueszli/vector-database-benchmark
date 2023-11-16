from google.cloud import dialogflow_v2

def sample_list_environments():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.EnvironmentsClient()
    request = dialogflow_v2.ListEnvironmentsRequest(parent='parent_value')
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)