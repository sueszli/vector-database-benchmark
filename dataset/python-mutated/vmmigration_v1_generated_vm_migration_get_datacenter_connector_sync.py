from google.cloud import vmmigration_v1

def sample_get_datacenter_connector():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetDatacenterConnectorRequest(name='name_value')
    response = client.get_datacenter_connector(request=request)
    print(response)