from google.cloud import dataplex_v1

def sample_update_data_scan():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataScanServiceClient()
    data_scan = dataplex_v1.DataScan()
    data_scan.data_quality_spec.rules.dimension = 'dimension_value'
    data_scan.data.entity = 'entity_value'
    request = dataplex_v1.UpdateDataScanRequest(data_scan=data_scan)
    operation = client.update_data_scan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)