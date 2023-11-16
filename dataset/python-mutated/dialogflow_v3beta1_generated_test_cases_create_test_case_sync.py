from google.cloud import dialogflowcx_v3beta1

def sample_create_test_case():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.TestCasesClient()
    test_case = dialogflowcx_v3beta1.TestCase()
    test_case.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.CreateTestCaseRequest(parent='parent_value', test_case=test_case)
    response = client.create_test_case(request=request)
    print(response)