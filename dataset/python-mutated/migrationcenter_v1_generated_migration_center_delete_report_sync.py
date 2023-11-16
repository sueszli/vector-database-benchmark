from google.cloud import migrationcenter_v1

def sample_delete_report():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteReportRequest(name='name_value')
    operation = client.delete_report(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)