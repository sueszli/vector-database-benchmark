from google.cloud import rapidmigrationassessment_v1

def sample_register_collector():
    if False:
        while True:
            i = 10
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.RegisterCollectorRequest(name='name_value')
    operation = client.register_collector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)