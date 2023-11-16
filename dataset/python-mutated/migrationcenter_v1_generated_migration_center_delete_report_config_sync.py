from google.cloud import migrationcenter_v1

def sample_delete_report_config():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteReportConfigRequest(name='name_value')
    operation = client.delete_report_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)