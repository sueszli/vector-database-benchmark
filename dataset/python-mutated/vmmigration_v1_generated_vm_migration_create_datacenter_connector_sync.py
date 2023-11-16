from google.cloud import vmmigration_v1

def sample_create_datacenter_connector():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateDatacenterConnectorRequest(parent='parent_value', datacenter_connector_id='datacenter_connector_id_value')
    operation = client.create_datacenter_connector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)