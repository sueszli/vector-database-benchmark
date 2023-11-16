from google.cloud import dialogflowcx_v3

def sample_get_test_case():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.GetTestCaseRequest(name='name_value')
    response = client.get_test_case(request=request)
    print(response)