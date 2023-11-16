from google.cloud import vmmigration_v1

def sample_upgrade_appliance():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.UpgradeApplianceRequest(datacenter_connector='datacenter_connector_value')
    operation = client.upgrade_appliance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)