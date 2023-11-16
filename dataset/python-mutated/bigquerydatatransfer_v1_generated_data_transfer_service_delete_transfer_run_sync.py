from google.cloud import bigquery_datatransfer_v1

def sample_delete_transfer_run():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.DeleteTransferRunRequest(name='name_value')
    client.delete_transfer_run(request=request)