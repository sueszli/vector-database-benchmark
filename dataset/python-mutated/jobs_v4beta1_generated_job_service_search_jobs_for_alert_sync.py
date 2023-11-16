from google.cloud import talent_v4beta1

def sample_search_jobs_for_alert():
    if False:
        while True:
            i = 10
    client = talent_v4beta1.JobServiceClient()
    request = talent_v4beta1.SearchJobsRequest(parent='parent_value')
    page_result = client.search_jobs_for_alert(request=request)
    for response in page_result:
        print(response)