from google.cloud import bigquery_datatransfer_v1

def sample_list_data_sources():
    if False:
        i = 10
        return i + 15
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.ListDataSourcesRequest(parent='parent_value')
    page_result = client.list_data_sources(request=request)
    for response in page_result:
        print(response)