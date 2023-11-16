from google.cloud import dataproc_v1

def sample_list_batches():
    if False:
        print('Hello World!')
    client = dataproc_v1.BatchControllerClient()
    request = dataproc_v1.ListBatchesRequest(parent='parent_value')
    page_result = client.list_batches(request=request)
    for response in page_result:
        print(response)