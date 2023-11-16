from google.cloud import dialogflowcx_v3

def sample_update_flow():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.FlowsClient()
    flow = dialogflowcx_v3.Flow()
    flow.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateFlowRequest(flow=flow)
    response = client.update_flow(request=request)
    print(response)