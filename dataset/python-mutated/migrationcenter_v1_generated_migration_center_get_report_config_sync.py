from google.cloud import migrationcenter_v1

def sample_get_report_config():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetReportConfigRequest(name='name_value')
    response = client.get_report_config(request=request)
    print(response)