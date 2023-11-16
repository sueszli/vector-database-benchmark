from google.cloud import rapidmigrationassessment_v1

def sample_get_annotation():
    if False:
        for i in range(10):
            print('nop')
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.GetAnnotationRequest(name='name_value')
    response = client.get_annotation(request=request)
    print(response)