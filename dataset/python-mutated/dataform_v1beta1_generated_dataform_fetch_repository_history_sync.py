from google.cloud import dataform_v1beta1

def sample_fetch_repository_history():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.FetchRepositoryHistoryRequest(name='name_value')
    page_result = client.fetch_repository_history(request=request)
    for response in page_result:
        print(response)