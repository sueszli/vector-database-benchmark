from google.cloud import dialogflowcx_v3

def sample_run_test_case():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.RunTestCaseRequest(name='name_value')
    operation = client.run_test_case(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)