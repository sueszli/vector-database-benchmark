from google.cloud import dataproc_v1

def sample_create_batch():
    if False:
        while True:
            i = 10
    client = dataproc_v1.BatchControllerClient()
    batch = dataproc_v1.Batch()
    batch.pyspark_batch.main_python_file_uri = 'main_python_file_uri_value'
    request = dataproc_v1.CreateBatchRequest(parent='parent_value', batch=batch)
    operation = client.create_batch(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)