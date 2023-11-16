from google.cloud import dialogflowcx_v3beta1

def sample_export_flow():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.FlowsClient()
    request = dialogflowcx_v3beta1.ExportFlowRequest(name='name_value')
    operation = client.export_flow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)