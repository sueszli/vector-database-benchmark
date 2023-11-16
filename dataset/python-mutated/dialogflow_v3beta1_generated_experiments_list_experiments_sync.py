from google.cloud import dialogflowcx_v3beta1

def sample_list_experiments():
    if False:
        return 10
    client = dialogflowcx_v3beta1.ExperimentsClient()
    request = dialogflowcx_v3beta1.ListExperimentsRequest(parent='parent_value')
    page_result = client.list_experiments(request=request)
    for response in page_result:
        print(response)