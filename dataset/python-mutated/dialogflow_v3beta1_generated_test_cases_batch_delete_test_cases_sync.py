from google.cloud import dialogflowcx_v3beta1

def sample_batch_delete_test_cases():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.BatchDeleteTestCasesRequest(parent='parent_value', names=['names_value1', 'names_value2'])
    client.batch_delete_test_cases(request=request)