from google.cloud import dialogflowcx_v3beta1

def sample_list_agents():
    if False:
        return 10
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.ListAgentsRequest(parent='parent_value')
    page_result = client.list_agents(request=request)
    for response in page_result:
        print(response)