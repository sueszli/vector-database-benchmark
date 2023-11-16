from google.cloud import vmmigration_v1

def sample_get_utilization_report():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetUtilizationReportRequest(name='name_value')
    response = client.get_utilization_report(request=request)
    print(response)