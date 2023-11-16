from google.cloud import talent_v4beta1

def sample_search_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.JobServiceClient()
    request = talent_v4beta1.SearchJobsRequest(parent='parent_value')
    page_result = client.search_jobs(request=request)
    for response in page_result:
        print(response)