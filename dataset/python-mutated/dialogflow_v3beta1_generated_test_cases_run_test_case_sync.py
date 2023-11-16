from google.cloud import dialogflowcx_v3beta1

def sample_run_test_case():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.RunTestCaseRequest(name='name_value')
    operation = client.run_test_case(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)