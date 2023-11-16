from google.cloud import dataplex_v1

def sample_create_data_scan():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataScanServiceClient()
    data_scan = dataplex_v1.DataScan()
    data_scan.data_quality_spec.rules.dimension = 'dimension_value'
    data_scan.data.entity = 'entity_value'
    request = dataplex_v1.CreateDataScanRequest(parent='parent_value', data_scan=data_scan, data_scan_id='data_scan_id_value')
    operation = client.create_data_scan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)