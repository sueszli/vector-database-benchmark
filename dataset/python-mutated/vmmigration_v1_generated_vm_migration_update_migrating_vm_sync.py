from google.cloud import vmmigration_v1

def sample_update_migrating_vm():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.UpdateMigratingVmRequest()
    operation = client.update_migrating_vm(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)