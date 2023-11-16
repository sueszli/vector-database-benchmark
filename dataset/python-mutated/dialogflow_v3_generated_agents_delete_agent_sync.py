from google.cloud import dialogflowcx_v3

def sample_delete_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.DeleteAgentRequest(name='name_value')
    client.delete_agent(request=request)