from google.cloud import dialogflowcx_v3

def sample_delete_flow():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.FlowsClient()
    request = dialogflowcx_v3.DeleteFlowRequest(name='name_value')
    client.delete_flow(request=request)