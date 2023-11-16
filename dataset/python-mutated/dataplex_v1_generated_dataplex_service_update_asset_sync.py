from google.cloud import dataplex_v1

def sample_update_asset():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataplexServiceClient()
    asset = dataplex_v1.Asset()
    asset.resource_spec.type_ = 'BIGQUERY_DATASET'
    request = dataplex_v1.UpdateAssetRequest(asset=asset)
    operation = client.update_asset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)