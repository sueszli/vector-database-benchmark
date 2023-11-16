from google.cloud import dialogflow_v2

def sample_set_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.AgentsClient()
    agent = dialogflow_v2.Agent()
    agent.parent = 'parent_value'
    agent.display_name = 'display_name_value'
    agent.default_language_code = 'default_language_code_value'
    agent.time_zone = 'time_zone_value'
    request = dialogflow_v2.SetAgentRequest(agent=agent)
    response = client.set_agent(request=request)
    print(response)