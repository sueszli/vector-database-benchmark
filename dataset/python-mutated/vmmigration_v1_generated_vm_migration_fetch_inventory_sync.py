from google.cloud import vmmigration_v1

def sample_fetch_inventory():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.FetchInventoryRequest(source='source_value')
    response = client.fetch_inventory(request=request)
    print(response)