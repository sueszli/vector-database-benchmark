from google.cloud import rapidmigrationassessment_v1

def sample_pause_collector():
    if False:
        print('Hello World!')
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.PauseCollectorRequest(name='name_value')
    operation = client.pause_collector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)