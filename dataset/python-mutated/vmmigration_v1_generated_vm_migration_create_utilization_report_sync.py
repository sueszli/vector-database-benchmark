from google.cloud import vmmigration_v1

def sample_create_utilization_report():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateUtilizationReportRequest(parent='parent_value', utilization_report_id='utilization_report_id_value')
    operation = client.create_utilization_report(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)