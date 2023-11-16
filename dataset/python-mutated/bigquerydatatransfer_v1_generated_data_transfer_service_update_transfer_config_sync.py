from google.cloud import bigquery_datatransfer_v1

def sample_update_transfer_config():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    transfer_config = bigquery_datatransfer_v1.TransferConfig()
    transfer_config.destination_dataset_id = 'destination_dataset_id_value'
    request = bigquery_datatransfer_v1.UpdateTransferConfigRequest(transfer_config=transfer_config)
    response = client.update_transfer_config(request=request)
    print(response)