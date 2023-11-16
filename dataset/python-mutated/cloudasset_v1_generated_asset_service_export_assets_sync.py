from google.cloud import asset_v1

def sample_export_assets():
    if False:
        while True:
            i = 10
    client = asset_v1.AssetServiceClient()
    output_config = asset_v1.OutputConfig()
    output_config.gcs_destination.uri = 'uri_value'
    request = asset_v1.ExportAssetsRequest(parent='parent_value', output_config=output_config)
    operation = client.export_assets(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)