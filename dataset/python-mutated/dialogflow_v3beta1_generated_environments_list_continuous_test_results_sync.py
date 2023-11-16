from google.cloud import dialogflowcx_v3beta1

def sample_list_continuous_test_results():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    request = dialogflowcx_v3beta1.ListContinuousTestResultsRequest(parent='parent_value')
    page_result = client.list_continuous_test_results(request=request)
    for response in page_result:
        print(response)