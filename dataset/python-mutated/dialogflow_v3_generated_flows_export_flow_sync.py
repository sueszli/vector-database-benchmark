from google.cloud import dialogflowcx_v3

def sample_export_flow():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.FlowsClient()
    request = dialogflowcx_v3.ExportFlowRequest(name='name_value')
    operation = client.export_flow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)