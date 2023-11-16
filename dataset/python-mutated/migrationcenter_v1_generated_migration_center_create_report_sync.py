from google.cloud import migrationcenter_v1

def sample_create_report():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.CreateReportRequest(parent='parent_value', report_id='report_id_value')
    operation = client.create_report(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)