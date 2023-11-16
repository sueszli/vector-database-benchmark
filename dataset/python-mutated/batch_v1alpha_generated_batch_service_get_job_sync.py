from google.cloud import batch_v1alpha

def sample_get_job():
    if False:
        print('Hello World!')
    client = batch_v1alpha.BatchServiceClient()
    request = batch_v1alpha.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)