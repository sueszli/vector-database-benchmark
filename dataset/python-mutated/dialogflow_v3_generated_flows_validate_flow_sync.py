from google.cloud import dialogflowcx_v3

def sample_validate_flow():
    if False:
        return 10
    client = dialogflowcx_v3.FlowsClient()
    request = dialogflowcx_v3.ValidateFlowRequest(name='name_value')
    response = client.validate_flow(request=request)
    print(response)