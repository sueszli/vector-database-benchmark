from google.cloud import deploy_v1

def sample_list_rollouts():
    if False:
        while True:
            i = 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListRolloutsRequest(parent='parent_value')
    page_result = client.list_rollouts(request=request)
    for response in page_result:
        print(response)