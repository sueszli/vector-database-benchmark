from google.cloud import scheduler_v1beta1

def sample_list_jobs():
    if False:
        print('Hello World!')
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.ListJobsRequest(parent='parent_value')
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)