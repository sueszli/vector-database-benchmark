from google.cloud import migrationcenter_v1

def sample_update_preference_set():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.UpdatePreferenceSetRequest()
    operation = client.update_preference_set(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)