from google.cloud import batch_v1alpha

def sample_list_jobs():
    if False:
        i = 10
        return i + 15
    client = batch_v1alpha.BatchServiceClient()
    request = batch_v1alpha.ListJobsRequest()
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)