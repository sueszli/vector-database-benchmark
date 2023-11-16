from google.cloud import rapidmigrationassessment_v1

def sample_update_collector():
    if False:
        return 10
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.UpdateCollectorRequest()
    operation = client.update_collector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)