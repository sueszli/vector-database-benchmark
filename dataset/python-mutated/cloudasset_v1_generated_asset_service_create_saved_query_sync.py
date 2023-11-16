from google.cloud import asset_v1

def sample_create_saved_query():
    if False:
        while True:
            i = 10
    client = asset_v1.AssetServiceClient()
    request = asset_v1.CreateSavedQueryRequest(parent='parent_value', saved_query_id='saved_query_id_value')
    response = client.create_saved_query(request=request)
    print(response)