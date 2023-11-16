from google.cloud import batch_v1

def sample_create_job():
    if False:
        while True:
            i = 10
    client = batch_v1.BatchServiceClient()
    request = batch_v1.CreateJobRequest(parent='parent_value')
    response = client.create_job(request=request)
    print(response)