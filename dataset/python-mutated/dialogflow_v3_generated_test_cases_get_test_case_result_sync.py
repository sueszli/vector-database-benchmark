from google.cloud import dialogflowcx_v3

def sample_get_test_case_result():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.GetTestCaseResultRequest(name='name_value')
    response = client.get_test_case_result(request=request)
    print(response)