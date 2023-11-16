from google.cloud import migrationcenter_v1

def sample_create_preference_set():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.CreatePreferenceSetRequest(parent='parent_value', preference_set_id='preference_set_id_value')
    operation = client.create_preference_set(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)