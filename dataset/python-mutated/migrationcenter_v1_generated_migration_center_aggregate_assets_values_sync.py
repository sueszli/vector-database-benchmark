from google.cloud import migrationcenter_v1

def sample_aggregate_assets_values():
    if False:
        print('Hello World!')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.AggregateAssetsValuesRequest(parent='parent_value')
    response = client.aggregate_assets_values(request=request)
    print(response)