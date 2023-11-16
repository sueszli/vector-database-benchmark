from google.cloud import rapidmigrationassessment_v1

def sample_resume_collector():
    if False:
        for i in range(10):
            print('nop')
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.ResumeCollectorRequest(name='name_value')
    operation = client.resume_collector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)