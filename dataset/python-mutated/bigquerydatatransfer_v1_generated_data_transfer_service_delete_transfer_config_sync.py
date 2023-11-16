from google.cloud import bigquery_datatransfer_v1

def sample_delete_transfer_config():
    if False:
        while True:
            i = 10
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.DeleteTransferConfigRequest(name='name_value')
    client.delete_transfer_config(request=request)