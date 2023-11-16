from google.cloud import batch_v1

def sample_delete_job():
    if False:
        i = 10
        return i + 15
    client = batch_v1.BatchServiceClient()
    request = batch_v1.DeleteJobRequest()
    operation = client.delete_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)