from google.cloud import dialogflowcx_v3beta1

def sample_list_test_case_results():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.TestCasesClient()
    request = dialogflowcx_v3beta1.ListTestCaseResultsRequest(parent='parent_value')
    page_result = client.list_test_case_results(request=request)
    for response in page_result:
        print(response)