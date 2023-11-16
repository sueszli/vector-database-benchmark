from google.cloud import talent_v4beta1

def sample_batch_delete_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.JobServiceClient()
    request = talent_v4beta1.BatchDeleteJobsRequest(parent='parent_value', filter='filter_value')
    client.batch_delete_jobs(request=request)