from google.cloud import rapidmigrationassessment_v1

def sample_get_collector():
    if False:
        for i in range(10):
            print('nop')
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.GetCollectorRequest(name='name_value')
    response = client.get_collector(request=request)
    print(response)