from google.cloud import run_v2

def sample_list_jobs():
    if False:
        i = 10
        return i + 15
    client = run_v2.JobsClient()
    request = run_v2.ListJobsRequest(parent='parent_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)