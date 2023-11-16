from google.cloud import bigquery_datatransfer_v1

def sample_start_manual_transfer_runs():
    if False:
        i = 10
        return i + 15
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.StartManualTransferRunsRequest()
    response = client.start_manual_transfer_runs(request=request)
    print(response)