from google.cloud import dataplex_v1

def sample_get_data_scan_job():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataScanServiceClient()
    request = dataplex_v1.GetDataScanJobRequest(name='name_value')
    response = client.get_data_scan_job(request=request)
    print(response)