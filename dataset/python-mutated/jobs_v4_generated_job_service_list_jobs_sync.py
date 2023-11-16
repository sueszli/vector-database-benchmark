from google.cloud import talent_v4

def sample_list_jobs():
    if False:
        i = 10
        return i + 15
    client = talent_v4.JobServiceClient()
    request = talent_v4.ListJobsRequest(parent='parent_value', filter='filter_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)