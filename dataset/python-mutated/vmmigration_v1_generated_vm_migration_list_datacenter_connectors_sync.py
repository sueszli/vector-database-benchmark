from google.cloud import vmmigration_v1

def sample_list_datacenter_connectors():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListDatacenterConnectorsRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_datacenter_connectors(request=request)
    for response in page_result:
        print(response)