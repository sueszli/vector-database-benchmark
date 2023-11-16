from google.cloud import vmmigration_v1

def sample_delete_migrating_vm():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.DeleteMigratingVmRequest(name='name_value')
    operation = client.delete_migrating_vm(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)