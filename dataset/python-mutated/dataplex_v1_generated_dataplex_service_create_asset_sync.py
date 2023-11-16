from google.cloud import dataplex_v1

def sample_create_asset():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataplexServiceClient()
    asset = dataplex_v1.Asset()
    asset.resource_spec.type_ = 'BIGQUERY_DATASET'
    request = dataplex_v1.CreateAssetRequest(parent='parent_value', asset_id='asset_id_value', asset=asset)
    operation = client.create_asset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)