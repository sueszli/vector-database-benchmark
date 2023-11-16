from google.cloud import dialogflowcx_v3beta1

def sample_delete_agent():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.DeleteAgentRequest(name='name_value')
    client.delete_agent(request=request)