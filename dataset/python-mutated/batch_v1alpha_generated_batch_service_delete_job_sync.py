from google.cloud import batch_v1alpha

def sample_delete_job():
    if False:
        return 10
    client = batch_v1alpha.BatchServiceClient()
    request = batch_v1alpha.DeleteJobRequest()
    operation = client.delete_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)