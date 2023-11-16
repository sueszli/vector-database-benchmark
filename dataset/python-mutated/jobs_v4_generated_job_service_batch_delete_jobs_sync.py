from google.cloud import talent_v4

def sample_batch_delete_jobs():
    if False:
        while True:
            i = 10
    client = talent_v4.JobServiceClient()
    request = talent_v4.BatchDeleteJobsRequest(parent='parent_value')
    operation = client.batch_delete_jobs(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)