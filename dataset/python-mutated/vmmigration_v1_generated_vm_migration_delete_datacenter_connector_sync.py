from google.cloud import vmmigration_v1

def sample_delete_datacenter_connector():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.DeleteDatacenterConnectorRequest(name='name_value')
    operation = client.delete_datacenter_connector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)