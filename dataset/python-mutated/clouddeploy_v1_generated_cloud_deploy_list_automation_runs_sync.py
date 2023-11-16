from google.cloud import deploy_v1

def sample_list_automation_runs():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListAutomationRunsRequest(parent='parent_value')
    page_result = client.list_automation_runs(request=request)
    for response in page_result:
        print(response)