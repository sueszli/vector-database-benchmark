from google.cloud import dialogflowcx_v3beta1

def sample_delete_flow():
    if False:
        return 10
    client = dialogflowcx_v3beta1.FlowsClient()
    request = dialogflowcx_v3beta1.DeleteFlowRequest(name='name_value')
    client.delete_flow(request=request)