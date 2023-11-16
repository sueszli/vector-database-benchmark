from google.cloud import dialogflowcx_v3

def sample_list_continuous_test_results():
    if False:
        return 10
    client = dialogflowcx_v3.EnvironmentsClient()
    request = dialogflowcx_v3.ListContinuousTestResultsRequest(parent='parent_value')
    page_result = client.list_continuous_test_results(request=request)
    for response in page_result:
        print(response)