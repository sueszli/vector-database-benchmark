from google.cloud import migrationcenter_v1

def sample_get_settings():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetSettingsRequest(name='name_value')
    response = client.get_settings(request=request)
    print(response)