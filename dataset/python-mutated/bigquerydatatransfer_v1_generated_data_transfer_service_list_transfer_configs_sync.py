from google.cloud import bigquery_datatransfer_v1

def sample_list_transfer_configs():
    if False:
        while True:
            i = 10
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.ListTransferConfigsRequest(parent='parent_value')
    page_result = client.list_transfer_configs(request=request)
    for response in page_result:
        print(response)