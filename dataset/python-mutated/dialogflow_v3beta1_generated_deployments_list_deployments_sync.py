from google.cloud import dialogflowcx_v3beta1

def sample_list_deployments():
    if False:
        return 10
    client = dialogflowcx_v3beta1.DeploymentsClient()
    request = dialogflowcx_v3beta1.ListDeploymentsRequest(parent='parent_value')
    page_result = client.list_deployments(request=request)
    for response in page_result:
        print(response)