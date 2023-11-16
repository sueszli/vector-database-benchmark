from google.cloud import dialogflowcx_v3beta1

def sample_validate_agent():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.ValidateAgentRequest(name='name_value')
    response = client.validate_agent(request=request)
    print(response)