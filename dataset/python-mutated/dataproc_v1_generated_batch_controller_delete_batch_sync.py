from google.cloud import dataproc_v1

def sample_delete_batch():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.BatchControllerClient()
    request = dataproc_v1.DeleteBatchRequest(name='name_value')
    client.delete_batch(request=request)