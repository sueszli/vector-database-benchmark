from google.cloud import migrationcenter_v1

def sample_create_report_config():
    if False:
        while True:
            i = 10
    client = migrationcenter_v1.MigrationCenterClient()
    report_config = migrationcenter_v1.ReportConfig()
    report_config.group_preferenceset_assignments.group = 'group_value'
    report_config.group_preferenceset_assignments.preference_set = 'preference_set_value'
    request = migrationcenter_v1.CreateReportConfigRequest(parent='parent_value', report_config_id='report_config_id_value', report_config=report_config)
    operation = client.create_report_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)