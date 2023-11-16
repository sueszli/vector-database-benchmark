from google.cloud import dialogflow_v2beta1

def sample_get_agent():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.AgentsClient()
    request = dialogflow_v2beta1.GetAgentRequest(parent='parent_value')
    response = client.get_agent(request=request)
    print(response)