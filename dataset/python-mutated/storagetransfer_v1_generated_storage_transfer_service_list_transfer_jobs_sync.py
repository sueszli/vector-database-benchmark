from google.cloud import storage_transfer_v1

def sample_list_transfer_jobs():
    if False:
        print('Hello World!')
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.ListTransferJobsRequest(filter='filter_value')
    page_result = client.list_transfer_jobs(request=request)
    for response in page_result:
        print(response)