from google.cloud import dataplex_v1

def sample_delete_data_scan():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.DataScanServiceClient()
    request = dataplex_v1.DeleteDataScanRequest(name='name_value')
    operation = client.delete_data_scan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)