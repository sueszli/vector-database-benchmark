from google.cloud import dataplex_v1

def sample_list_data_scan_jobs():
    if False:
        return 10
    client = dataplex_v1.DataScanServiceClient()
    request = dataplex_v1.ListDataScanJobsRequest(parent='parent_value')
    page_result = client.list_data_scan_jobs(request=request)
    for response in page_result:
        print(response)