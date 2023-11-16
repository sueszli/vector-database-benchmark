from google.cloud import bigquery_datatransfer_v1

def sample_check_valid_creds():
    if False:
        i = 10
        return i + 15
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.CheckValidCredsRequest(name='name_value')
    response = client.check_valid_creds(request=request)
    print(response)