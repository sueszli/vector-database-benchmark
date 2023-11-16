from google.cloud import migrationcenter_v1

def sample_get_report():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetReportRequest(name='name_value')
    response = client.get_report(request=request)
    print(response)