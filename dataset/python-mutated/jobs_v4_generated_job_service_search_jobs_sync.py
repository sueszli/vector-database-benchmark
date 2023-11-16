from google.cloud import talent_v4

def sample_search_jobs():
    if False:
        while True:
            i = 10
    client = talent_v4.JobServiceClient()
    request = talent_v4.SearchJobsRequest(parent='parent_value')
    response = client.search_jobs(request=request)
    print(response)