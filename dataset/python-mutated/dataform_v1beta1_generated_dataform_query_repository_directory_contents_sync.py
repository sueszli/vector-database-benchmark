from google.cloud import dataform_v1beta1

def sample_query_repository_directory_contents():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.QueryRepositoryDirectoryContentsRequest(name='name_value')
    page_result = client.query_repository_directory_contents(request=request)
    for response in page_result:
        print(response)