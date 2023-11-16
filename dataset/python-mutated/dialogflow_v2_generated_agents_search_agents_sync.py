from google.cloud import dialogflow_v2

def sample_search_agents():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.SearchAgentsRequest(parent='parent_value')
    page_result = client.search_agents(request=request)
    for response in page_result:
        print(response)