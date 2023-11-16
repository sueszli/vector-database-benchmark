from google.cloud import assuredworkloads_v1

def sample_list_workloads():
    if False:
        i = 10
        return i + 15
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1.ListWorkloadsRequest(parent='parent_value')
    page_result = client.list_workloads(request=request)
    for response in page_result:
        print(response)