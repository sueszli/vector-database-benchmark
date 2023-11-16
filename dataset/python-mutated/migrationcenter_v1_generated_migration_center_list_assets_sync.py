from google.cloud import migrationcenter_v1

def sample_list_assets():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ListAssetsRequest(parent='parent_value')
    page_result = client.list_assets(request=request)
    for response in page_result:
        print(response)