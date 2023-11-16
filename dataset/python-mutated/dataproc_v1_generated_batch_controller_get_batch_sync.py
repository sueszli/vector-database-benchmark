from google.cloud import dataproc_v1

def sample_get_batch():
    if False:
        print('Hello World!')
    client = dataproc_v1.BatchControllerClient()
    request = dataproc_v1.GetBatchRequest(name='name_value')
    response = client.get_batch(request=request)
    print(response)