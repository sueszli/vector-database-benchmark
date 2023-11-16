from google.cloud import dialogflowcx_v3beta1

def sample_get_flow_validation_result():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.FlowsClient()
    request = dialogflowcx_v3beta1.GetFlowValidationResultRequest(name='name_value')
    response = client.get_flow_validation_result(request=request)
    print(response)