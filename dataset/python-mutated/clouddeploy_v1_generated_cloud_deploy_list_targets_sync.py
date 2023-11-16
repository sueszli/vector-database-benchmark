from google.cloud import deploy_v1

def sample_list_targets():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListTargetsRequest(parent='parent_value')
    page_result = client.list_targets(request=request)
    for response in page_result:
        print(response)