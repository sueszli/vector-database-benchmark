from google.cloud import dialogflowcx_v3beta1

def sample_import_test_cases():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.ImportTestCasesRequest(gcs_uri='gcs_uri_value', parent='parent_value')
    operation = client.import_test_cases(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)