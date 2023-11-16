from google.cloud import bigquery_datatransfer_v1

def sample_list_transfer_runs():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.ListTransferRunsRequest(parent='parent_value')
    page_result = client.list_transfer_runs(request=request)
    for response in page_result:
        print(response)