from google.cloud import rapidmigrationassessment_v1

def sample_create_annotation():
    if False:
        return 10
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.CreateAnnotationRequest(parent='parent_value')
    operation = client.create_annotation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)