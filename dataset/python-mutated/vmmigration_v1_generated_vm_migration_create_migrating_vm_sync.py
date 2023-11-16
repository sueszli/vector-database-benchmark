from google.cloud import vmmigration_v1

def sample_create_migrating_vm():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateMigratingVmRequest(parent='parent_value', migrating_vm_id='migrating_vm_id_value')
    operation = client.create_migrating_vm(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)