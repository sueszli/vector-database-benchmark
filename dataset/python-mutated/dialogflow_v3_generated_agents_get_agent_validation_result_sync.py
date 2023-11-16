from google.cloud import dialogflowcx_v3

def sample_get_agent_validation_result():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.GetAgentValidationResultRequest(name='name_value')
    response = client.get_agent_validation_result(request=request)
    print(response)