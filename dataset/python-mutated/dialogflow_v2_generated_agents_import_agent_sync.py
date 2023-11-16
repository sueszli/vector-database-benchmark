from google.cloud import dialogflow_v2

def sample_import_agent():
    if False:
        print('Hello World!')
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.ImportAgentRequest(agent_uri='agent_uri_value', parent='parent_value')
    operation = client.import_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)