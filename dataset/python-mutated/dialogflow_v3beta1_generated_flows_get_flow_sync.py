from google.cloud import dialogflowcx_v3beta1

def sample_get_flow():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.FlowsClient()
    request = dialogflowcx_v3beta1.GetFlowRequest(name='name_value')
    response = client.get_flow(request=request)
    print(response)