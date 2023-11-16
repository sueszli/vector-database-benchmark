from google.cloud import dialogflowcx_v3

def sample_batch_run_test_cases():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.BatchRunTestCasesRequest(parent='parent_value', test_cases=['test_cases_value1', 'test_cases_value2'])
    operation = client.batch_run_test_cases(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)