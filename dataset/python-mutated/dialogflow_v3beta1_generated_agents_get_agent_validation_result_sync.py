from google.cloud import dialogflowcx_v3beta1

def sample_get_agent_validation_result():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.GetAgentValidationResultRequest(name='name_value')
    response = client.get_agent_validation_result(request=request)
    print(response)