from google.cloud import dialogflowcx_v3beta1

def sample_export_test_cases():
    if False:
        return 10
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.ExportTestCasesRequest(gcs_uri='gcs_uri_value', parent='parent_value')
    operation = client.export_test_cases(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)