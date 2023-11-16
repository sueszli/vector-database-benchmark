from google.cloud import vmmigration_v1

def sample_list_replication_cycles():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListReplicationCyclesRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_replication_cycles(request=request)
    for response in page_result:
        print(response)