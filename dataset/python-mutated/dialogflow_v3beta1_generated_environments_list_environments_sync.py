from google.cloud import dialogflowcx_v3beta1

def sample_list_environments():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    request = dialogflowcx_v3beta1.ListEnvironmentsRequest(parent='parent_value')
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)