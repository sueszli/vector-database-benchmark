from google.cloud import vmmigration_v1

def sample_delete_utilization_report():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.DeleteUtilizationReportRequest(name='name_value')
    operation = client.delete_utilization_report(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)