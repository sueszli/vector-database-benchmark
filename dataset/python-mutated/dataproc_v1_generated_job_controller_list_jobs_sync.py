from google.cloud import dataproc_v1

def sample_list_jobs():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.JobControllerClient()
    request = dataproc_v1.ListJobsRequest(project_id='project_id_value', region='region_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)