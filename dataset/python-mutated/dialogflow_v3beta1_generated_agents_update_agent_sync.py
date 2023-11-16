from google.cloud import dialogflowcx_v3beta1

def sample_update_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.AgentsClient()
    agent = dialogflowcx_v3beta1.Agent()
    agent.display_name = 'display_name_value'
    agent.default_language_code = 'default_language_code_value'
    agent.time_zone = 'time_zone_value'
    request = dialogflowcx_v3beta1.UpdateAgentRequest(agent=agent)
    response = client.update_agent(request=request)
    print(response)