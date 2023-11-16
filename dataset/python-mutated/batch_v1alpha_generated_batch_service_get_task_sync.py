from google.cloud import batch_v1alpha

def sample_get_task():
    if False:
        i = 10
        return i + 15
    client = batch_v1alpha.BatchServiceClient()
    request = batch_v1alpha.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)