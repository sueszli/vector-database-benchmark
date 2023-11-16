from google.cloud import dialogflowcx_v3

def sample_update_test_case():
    if False:
        return 10
    client = dialogflowcx_v3.TestCasesClient()
    test_case = dialogflowcx_v3.TestCase()
    test_case.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateTestCaseRequest(test_case=test_case)
    response = client.update_test_case(request=request)
    print(response)