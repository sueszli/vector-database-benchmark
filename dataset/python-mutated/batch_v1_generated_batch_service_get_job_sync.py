from google.cloud import batch_v1

def sample_get_job():
    if False:
        return 10
    client = batch_v1.BatchServiceClient()
    request = batch_v1.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)