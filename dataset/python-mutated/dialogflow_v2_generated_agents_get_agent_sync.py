from google.cloud import dialogflow_v2

def sample_get_agent():
    if False:
        print('Hello World!')
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.GetAgentRequest(parent='parent_value')
    response = client.get_agent(request=request)
    print(response)