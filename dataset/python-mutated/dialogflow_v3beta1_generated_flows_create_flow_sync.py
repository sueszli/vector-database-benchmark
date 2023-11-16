from google.cloud import dialogflowcx_v3beta1

def sample_create_flow():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.FlowsClient()
    flow = dialogflowcx_v3beta1.Flow()
    flow.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.CreateFlowRequest(parent='parent_value', flow=flow)
    response = client.create_flow(request=request)
    print(response)