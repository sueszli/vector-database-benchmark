from google.cloud import dataplex_v1

def sample_get_data_scan():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataScanServiceClient()
    request = dataplex_v1.GetDataScanRequest(name='name_value')
    response = client.get_data_scan(request=request)
    print(response)