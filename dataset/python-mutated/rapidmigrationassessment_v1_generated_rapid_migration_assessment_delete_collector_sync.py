from google.cloud import rapidmigrationassessment_v1

def sample_delete_collector():
    if False:
        for i in range(10):
            print('nop')
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.DeleteCollectorRequest(name='name_value')
    operation = client.delete_collector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)