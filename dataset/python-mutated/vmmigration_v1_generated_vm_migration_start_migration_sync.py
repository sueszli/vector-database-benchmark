from google.cloud import vmmigration_v1

def sample_start_migration():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.StartMigrationRequest(migrating_vm='migrating_vm_value')
    operation = client.start_migration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)