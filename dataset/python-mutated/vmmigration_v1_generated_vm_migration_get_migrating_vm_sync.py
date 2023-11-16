from google.cloud import vmmigration_v1

def sample_get_migrating_vm():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetMigratingVmRequest(name='name_value')
    response = client.get_migrating_vm(request=request)
    print(response)