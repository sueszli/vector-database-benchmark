from google.cloud import bigquery_datatransfer_v1

def sample_schedule_transfer_runs():
    if False:
        print('Hello World!')
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.ScheduleTransferRunsRequest(parent='parent_value')
    response = client.schedule_transfer_runs(request=request)
    print(response)