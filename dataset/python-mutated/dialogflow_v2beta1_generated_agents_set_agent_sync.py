from google.cloud import dialogflow_v2beta1

def sample_set_agent():
    if False:
        return 10
    client = dialogflow_v2beta1.AgentsClient()
    agent = dialogflow_v2beta1.Agent()
    agent.parent = 'parent_value'
    request = dialogflow_v2beta1.SetAgentRequest(agent=agent)
    response = client.set_agent(request=request)
    print(response)