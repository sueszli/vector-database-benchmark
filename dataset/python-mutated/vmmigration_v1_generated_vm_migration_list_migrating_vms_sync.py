from google.cloud import vmmigration_v1

def sample_list_migrating_vms():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListMigratingVmsRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_migrating_vms(request=request)
    for response in page_result:
        print(response)