from google.cloud import dialogflowcx_v3

def sample_batch_delete_test_cases():
    if False:
        return 10
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.BatchDeleteTestCasesRequest(parent='parent_value', names=['names_value1', 'names_value2'])
    client.batch_delete_test_cases(request=request)