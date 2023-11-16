from google.cloud import vmmigration_v1

def sample_pause_migration():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.PauseMigrationRequest(migrating_vm='migrating_vm_value')
    operation = client.pause_migration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)