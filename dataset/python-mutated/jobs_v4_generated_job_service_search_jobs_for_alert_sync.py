from google.cloud import talent_v4

def sample_search_jobs_for_alert():
    if False:
        i = 10
        return i + 15
    client = talent_v4.JobServiceClient()
    request = talent_v4.SearchJobsRequest(parent='parent_value')
    response = client.search_jobs_for_alert(request=request)
    print(response)