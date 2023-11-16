from google.cloud import talent_v4beta1

def sample_list_jobs():
    if False:
        while True:
            i = 10
    client = talent_v4beta1.JobServiceClient()
    request = talent_v4beta1.ListJobsRequest(parent='parent_value', filter='filter_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)