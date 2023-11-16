from google.cloud import dialogflowcx_v3

def sample_list_deployments():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.DeploymentsClient()
    request = dialogflowcx_v3.ListDeploymentsRequest(parent='parent_value')
    page_result = client.list_deployments(request=request)
    for response in page_result:
        print(response)