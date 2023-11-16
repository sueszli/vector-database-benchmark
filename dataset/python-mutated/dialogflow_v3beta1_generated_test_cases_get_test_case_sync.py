from google.cloud import dialogflowcx_v3beta1

def sample_get_test_case():
    if False:
        return 10
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.GetTestCaseRequest(name='name_value')
    response = client.get_test_case(request=request)
    print(response)