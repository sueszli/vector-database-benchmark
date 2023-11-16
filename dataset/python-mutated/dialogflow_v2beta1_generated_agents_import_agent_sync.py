from google.cloud import dialogflow_v2beta1

def sample_import_agent():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.AgentsClient()
    request = dialogflow_v2beta1.ImportAgentRequest(agent_uri='agent_uri_value', parent='parent_value')
    operation = client.import_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)