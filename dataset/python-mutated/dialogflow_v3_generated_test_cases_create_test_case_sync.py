from google.cloud import dialogflowcx_v3

def sample_create_test_case():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.TestCasesClient()
    test_case = dialogflowcx_v3.TestCase()
    test_case.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreateTestCaseRequest(parent='parent_value', test_case=test_case)
    response = client.create_test_case(request=request)
    print(response)