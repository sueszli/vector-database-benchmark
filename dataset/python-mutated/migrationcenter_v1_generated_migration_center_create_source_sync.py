from google.cloud import migrationcenter_v1

def sample_create_source():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.CreateSourceRequest(parent='parent_value', source_id='source_id_value')
    operation = client.create_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)