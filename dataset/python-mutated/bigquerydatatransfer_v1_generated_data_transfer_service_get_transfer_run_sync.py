from google.cloud import bigquery_datatransfer_v1

def sample_get_transfer_run():
    if False:
        return 10
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.GetTransferRunRequest(name='name_value')
    response = client.get_transfer_run(request=request)
    print(response)