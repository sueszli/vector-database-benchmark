from google.cloud import dialogflowcx_v3

def sample_list_test_case_results():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.TestCasesClient()
    request = dialogflowcx_v3.ListTestCaseResultsRequest(parent='parent_value')
    page_result = client.list_test_case_results(request=request)
    for response in page_result:
        print(response)