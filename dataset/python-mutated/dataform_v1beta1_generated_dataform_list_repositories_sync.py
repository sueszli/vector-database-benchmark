from google.cloud import dataform_v1beta1

def sample_list_repositories():
    if False:
        print('Hello World!')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ListRepositoriesRequest(parent='parent_value')
    page_result = client.list_repositories(request=request)
    for response in page_result:
        print(response)