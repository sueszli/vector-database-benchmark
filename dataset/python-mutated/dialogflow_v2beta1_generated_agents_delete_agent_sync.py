from google.cloud import dialogflow_v2beta1

def sample_delete_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.AgentsClient()
    request = dialogflow_v2beta1.DeleteAgentRequest(parent='parent_value')
    client.delete_agent(request=request)