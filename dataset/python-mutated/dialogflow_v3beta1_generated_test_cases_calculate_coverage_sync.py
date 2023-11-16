from google.cloud import dialogflowcx_v3beta1

def sample_calculate_coverage():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.CalculateCoverageRequest(agent='agent_value', type_='TRANSITION_ROUTE_GROUP')
    response = client.calculate_coverage(request=request)
    print(response)