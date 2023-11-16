from google.cloud import migrationcenter_v1

def sample_update_settings():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.UpdateSettingsRequest()
    operation = client.update_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)