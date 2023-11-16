from google.cloud import migrationcenter_v1

def sample_report_asset_frames():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ReportAssetFramesRequest(parent='parent_value', source='source_value')
    response = client.report_asset_frames(request=request)
    print(response)