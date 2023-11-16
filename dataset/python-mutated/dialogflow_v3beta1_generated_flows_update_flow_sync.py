from google.cloud import dialogflowcx_v3beta1

def sample_update_flow():
    if False:
        return 10
    client = dialogflowcx_v3beta1.FlowsClient()
    flow = dialogflowcx_v3beta1.Flow()
    flow.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateFlowRequest(flow=flow)
    response = client.update_flow(request=request)
    print(response)