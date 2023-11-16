from google.cloud import dataplex_v1

def sample_run_data_scan():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataScanServiceClient()
    request = dataplex_v1.RunDataScanRequest(name='name_value')
    response = client.run_data_scan(request=request)
    print(response)