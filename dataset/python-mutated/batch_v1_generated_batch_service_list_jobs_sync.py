from google.cloud import batch_v1

def sample_list_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = batch_v1.BatchServiceClient()
    request = batch_v1.ListJobsRequest()
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)