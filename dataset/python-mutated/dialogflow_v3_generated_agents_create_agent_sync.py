from google.cloud import dialogflowcx_v3

def sample_create_agent():
    if False:
        return 10
    client = dialogflowcx_v3.AgentsClient()
    agent = dialogflowcx_v3.Agent()
    agent.display_name = 'display_name_value'
    agent.default_language_code = 'default_language_code_value'
    agent.time_zone = 'time_zone_value'
    request = dialogflowcx_v3.CreateAgentRequest(parent='parent_value', agent=agent)
    response = client.create_agent(request=request)
    print(response)