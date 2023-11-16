from google.cloud import batch_v1alpha

def sample_create_job():
    if False:
        return 10
    client = batch_v1alpha.BatchServiceClient()
    request = batch_v1alpha.CreateJobRequest(parent='parent_value')
    response = client.create_job(request=request)
    print(response)