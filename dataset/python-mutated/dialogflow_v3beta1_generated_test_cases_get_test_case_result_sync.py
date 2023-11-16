from google.cloud import dialogflowcx_v3beta1

def sample_get_test_case_result():
    if False:
        return 10
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.GetTestCaseResultRequest(name='name_value')
    response = client.get_test_case_result(request=request)
    print(response)