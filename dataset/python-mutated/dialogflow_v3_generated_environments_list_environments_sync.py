from google.cloud import dialogflowcx_v3

def sample_list_environments():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.EnvironmentsClient()
    request = dialogflowcx_v3.ListEnvironmentsRequest(parent='parent_value')
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)