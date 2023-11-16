from google.cloud import dialogflowcx_v3

def sample_calculate_coverage():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.CalculateCoverageRequest(agent='agent_value', type_='TRANSITION_ROUTE_GROUP')
    response = client.calculate_coverage(request=request)
    print(response)