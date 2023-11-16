from google.cloud import bigquery_datatransfer_v1

def sample_get_data_source():
    if False:
        return 10
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.GetDataSourceRequest(name='name_value')
    response = client.get_data_source(request=request)
    print(response)