from google.cloud import dialogflowcx_v3beta1

def sample_get_agent():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.GetAgentRequest(name='name_value')
    response = client.get_agent(request=request)
    print(response)