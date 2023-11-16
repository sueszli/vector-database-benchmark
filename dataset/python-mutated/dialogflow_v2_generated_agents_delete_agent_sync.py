from google.cloud import dialogflow_v2

def sample_delete_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.DeleteAgentRequest(parent='parent_value')
    client.delete_agent(request=request)