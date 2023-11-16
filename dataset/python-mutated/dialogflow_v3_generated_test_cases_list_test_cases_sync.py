from google.cloud import dialogflowcx_v3

def sample_list_test_cases():
    if False:
        return 10
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.ListTestCasesRequest(parent='parent_value')
    page_result = client.list_test_cases(request=request)
    for response in page_result:
        print(response)