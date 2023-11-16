from google.cloud import deploy_v1

def sample_list_releases():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListReleasesRequest(parent='parent_value')
    page_result = client.list_releases(request=request)
    for response in page_result:
        print(response)