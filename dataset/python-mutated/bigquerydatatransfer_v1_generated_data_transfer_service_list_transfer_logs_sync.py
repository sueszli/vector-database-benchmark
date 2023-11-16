from google.cloud import bigquery_datatransfer_v1

def sample_list_transfer_logs():
    if False:
        return 10
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.ListTransferLogsRequest(parent='parent_value')
    page_result = client.list_transfer_logs(request=request)
    for response in page_result:
        print(response)