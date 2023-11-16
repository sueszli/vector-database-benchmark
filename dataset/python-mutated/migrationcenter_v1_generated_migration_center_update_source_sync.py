from google.cloud import migrationcenter_v1

def sample_update_source():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.UpdateSourceRequest()
    operation = client.update_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)