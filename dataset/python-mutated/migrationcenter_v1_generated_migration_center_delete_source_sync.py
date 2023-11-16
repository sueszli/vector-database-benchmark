from google.cloud import migrationcenter_v1

def sample_delete_source():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteSourceRequest(name='name_value')
    operation = client.delete_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)