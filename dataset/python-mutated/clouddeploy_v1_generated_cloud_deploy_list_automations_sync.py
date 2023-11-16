from google.cloud import deploy_v1

def sample_list_automations():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListAutomationsRequest(parent='parent_value')
    page_result = client.list_automations(request=request)
    for response in page_result:
        print(response)