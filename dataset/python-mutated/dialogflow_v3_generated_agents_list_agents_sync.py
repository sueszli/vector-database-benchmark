from google.cloud import dialogflowcx_v3

def sample_list_agents():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.ListAgentsRequest(parent='parent_value')
    page_result = client.list_agents(request=request)
    for response in page_result:
        print(response)