from google.cloud import bigquery_datatransfer_v1

def sample_get_transfer_config():
    if False:
        i = 10
        return i + 15
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.GetTransferConfigRequest(name='name_value')
    response = client.get_transfer_config(request=request)
    print(response)