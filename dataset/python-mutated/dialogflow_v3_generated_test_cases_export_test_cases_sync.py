from google.cloud import dialogflowcx_v3

def sample_export_test_cases():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.ExportTestCasesRequest(gcs_uri='gcs_uri_value', parent='parent_value')
    operation = client.export_test_cases(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)