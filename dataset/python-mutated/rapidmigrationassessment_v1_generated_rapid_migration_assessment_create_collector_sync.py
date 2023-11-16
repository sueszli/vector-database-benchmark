from google.cloud import rapidmigrationassessment_v1

def sample_create_collector():
    if False:
        i = 10
        return i + 15
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.CreateCollectorRequest(parent='parent_value', collector_id='collector_id_value')
    operation = client.create_collector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)