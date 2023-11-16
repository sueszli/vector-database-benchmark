from google.cloud import dialogflowcx_v3

def sample_get_flow():
    if False:
        return 10
    client = dialogflowcx_v3.FlowsClient()
    request = dialogflowcx_v3.GetFlowRequest(name='name_value')
    response = client.get_flow(request=request)
    print(response)