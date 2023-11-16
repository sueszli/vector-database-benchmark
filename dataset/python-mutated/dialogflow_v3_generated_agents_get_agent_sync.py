from google.cloud import dialogflowcx_v3

def sample_get_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.GetAgentRequest(name='name_value')
    response = client.get_agent(request=request)
    print(response)