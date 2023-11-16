from google.cloud import dialogflowcx_v3beta1

def sample_validate_flow():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.FlowsClient()
    request = dialogflowcx_v3beta1.ValidateFlowRequest(name='name_value')
    response = client.validate_flow(request=request)
    print(response)