from google.cloud import migrationcenter_v1

def sample_get_preference_set():
    if False:
        print('Hello World!')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetPreferenceSetRequest(name='name_value')
    response = client.get_preference_set(request=request)
    print(response)