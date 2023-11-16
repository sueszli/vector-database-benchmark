from google.cloud import batch_v1

def sample_get_task():
    if False:
        return 10
    client = batch_v1.BatchServiceClient()
    request = batch_v1.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)