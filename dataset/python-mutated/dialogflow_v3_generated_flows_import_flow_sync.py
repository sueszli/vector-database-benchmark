from google.cloud import dialogflowcx_v3

def sample_import_flow():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.FlowsClient()
    request = dialogflowcx_v3.ImportFlowRequest(flow_uri='flow_uri_value', parent='parent_value')
    operation = client.import_flow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)