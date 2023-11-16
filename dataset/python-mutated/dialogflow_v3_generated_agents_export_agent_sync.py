from google.cloud import dialogflowcx_v3

def sample_export_agent():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.ExportAgentRequest(name='name_value')
    operation = client.export_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)