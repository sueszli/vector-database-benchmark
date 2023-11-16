from google.cloud.devtools import cloudbuild_v1

def sample_list_worker_pools():
    if False:
        print('Hello World!')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.ListWorkerPoolsRequest(parent='parent_value')
    page_result = client.list_worker_pools(request=request)
    for response in page_result:
        print(response)