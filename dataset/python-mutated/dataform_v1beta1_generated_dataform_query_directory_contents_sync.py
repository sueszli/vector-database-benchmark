from google.cloud import dataform_v1beta1

def sample_query_directory_contents():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.QueryDirectoryContentsRequest(workspace='workspace_value')
    page_result = client.query_directory_contents(request=request)
    for response in page_result:
        print(response)