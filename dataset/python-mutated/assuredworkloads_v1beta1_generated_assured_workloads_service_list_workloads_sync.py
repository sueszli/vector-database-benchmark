from google.cloud import assuredworkloads_v1beta1

def sample_list_workloads():
    if False:
        print('Hello World!')
    client = assuredworkloads_v1beta1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1beta1.ListWorkloadsRequest(parent='parent_value')
    page_result = client.list_workloads(request=request)
    for response in page_result:
        print(response)