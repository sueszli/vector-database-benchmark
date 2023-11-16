from google.cloud import vmmigration_v1

def sample_finalize_migration():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.FinalizeMigrationRequest(migrating_vm='migrating_vm_value')
    operation = client.finalize_migration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)