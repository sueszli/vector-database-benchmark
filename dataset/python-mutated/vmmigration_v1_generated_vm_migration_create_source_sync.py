from google.cloud import vmmigration_v1

def sample_create_source():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateSourceRequest(parent='parent_value', source_id='source_id_value')
    operation = client.create_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)