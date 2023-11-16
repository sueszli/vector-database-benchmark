from google.cloud import gsuiteaddons_v1

def sample_list_deployments():
    if False:
        for i in range(10):
            print('nop')
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.ListDeploymentsRequest(parent='parent_value')
    page_result = client.list_deployments(request=request)
    for response in page_result:
        print(response)