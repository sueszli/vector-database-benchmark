from google.cloud import dialogflowcx_v3

def sample_validate_agent():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.ValidateAgentRequest(name='name_value')
    response = client.validate_agent(request=request)
    print(response)